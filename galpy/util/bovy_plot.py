##############################################################################
#
#   bovy_plot.py: general wrappers for matplotlib plotting
#
#       'public' methods:
#                         bovy_end_print
#                         bovy_dens2d
#                         bovy_hist
#                         bovy_plot
#                         bovy_print
#                         scatterplot (like hogg_scatterplot)
#                         bovy_text
#
#                         this module also defines a custom matplotlib 
#                         projection in which the polar azimuth increases
#                         clockwise (as in, the Galaxy viewed from the NGP)
#                         
#############################################################################
#############################################################################
#Copyright (c) 2010 - 2013, Jo Bovy
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import re
import math as m
import scipy as sc
from scipy import special
from scipy import interpolate
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import NullFormatter
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform
from mpl_toolkits.mplot3d import Axes3D
from galpy.util.config import __config__
if __config__.getboolean('plot','seaborn-bovy-defaults'):
    try:
        import seaborn as sns
    except: pass
    else:
        sns.set_style('ticks',
                      {'xtick.direction': u'in',
                       'ytick.direction': u'in',
                       'axes.labelsize': 18.0,
                       'axes.titlesize': 18.0,
                       'figure.figsize': sc.array([ 6.64,  4.  ]),
                       'grid.linewidth': 2.0,
                       'legend.fontsize': 18.0,
                       'lines.linewidth': 2.0,
                       'lines.markeredgewidth': 0.0,
                       'lines.markersize': 14.0,
                       'patch.linewidth': 0.6,
                       'xtick.labelsize': 16.0,
                       'xtick.major.pad': 14.0,
                       'xtick.major.width': 2.0,
                       'xtick.minor.width': 1.0,
                       'ytick.labelsize': 16.0,
                       'ytick.major.pad': 14.0,
                       'ytick.major.width': 2.0,})
_DEFAULTNCNTR= 10
def bovy_end_print(filename,**kwargs):
    """
    NAME:

       bovy_end_print

    PURPOSE:

       saves the current figure(s) to filename

    INPUT:

       filename - filename for plot (with extension)

    OPTIONAL INPUTS:

       format - file-format

    OUTPUT:

       (none)

    HISTORY:

       2009-12-23 - Written - Bovy (NYU)

    """
    if 'format' in kwargs:
        pyplot.savefig(filename,**kwargs)
    else:
        pyplot.savefig(filename,format=re.split(r'\.',filename)[-1],**kwargs)
    pyplot.close()

def bovy_hist(x,xlabel=None,ylabel=None,overplot=False,**kwargs):
    """
    NAME:

       bovy_hist

    PURPOSE:

       wrapper around matplotlib's hist function

    INPUT:

       x - array to histogram

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       yrange - set the y-axis range

       +all pyplot.hist keywords

    OUTPUT:
       (from the matplotlib docs:
       http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.hist)

       The return value is a tuple (n, bins, patches)
       or ([n0, n1, ...], bins, [patches0, patches1,...])
       if the input contains multiple data

    HISTORY:

       2009-12-23 - Written - Bovy (NYU)

    """
    if not overplot:
        pyplot.figure()
    if 'xrange' in kwargs:
        xlimits= kwargs.pop('xrange')
        if not 'range' in kwargs:
            kwargs['range']= xlimits
        xrangeSet= True
    else: xrangeSet= False
    if 'yrange' in kwargs:
        ylimits= kwargs.pop('yrange')
        yrangeSet= True
    else: yrangeSet= False
    out= pyplot.hist(x,**kwargs)
    if overplot: return out
    _add_axislabels(xlabel,ylabel)
    if not 'range' in kwargs and not xrangeSet:
        if isinstance(x,list):
            xlimits=(sc.array(x).min(),sc.array(x).max())
        else:
            pyplot.xlim(x.min(),x.max())
    elif xrangeSet:
        pyplot.xlim(xlimits)
    else:
        pyplot.xlim(kwargs['range'])
    if yrangeSet:
        pyplot.ylim(ylimits)
    _add_ticks()
    return out

def bovy_plot(*args,**kwargs):
    """
    NAME:

       bovy_plot

    PURPOSE:

       wrapper around matplotlib's plot function

    INPUT:

       see http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       xrange

       yrange

       scatter= if True, use pyplot.scatter and its options etc.

       colorbar= if True, and scatter==True, add colorbar

       crange - range for colorbar of scatter==True

       clabel= label for colorbar

       overplot=True does not start a new figure and does not change the ranges and labels

       gcf=True does not start a new figure (does change the ranges and labels)

       onedhists - if True, make one-d histograms on the sides

       onedhistcolor, onedhistfc, onedhistec

       onedhistxnormed, onedhistynormed - normed keyword for one-d histograms
       
       onedhistxweights, onedhistyweights - weights keyword for one-d histograms

       bins= number of bins for onedhists

       semilogx=, semilogy=, loglog= if True, plot logs

    OUTPUT:

       plot to output device, returns what pyplot.plot returns, or 3 Axes instances if onedhists=True

    HISTORY:

       2009-12-28 - Written - Bovy (NYU)

    """
    overplot= kwargs.pop('overplot',False)
    gcf= kwargs.pop('gcf',False)
    onedhists= kwargs.pop('onedhists',False)
    scatter= kwargs.pop('scatter',False)
    loglog= kwargs.pop('loglog',False)
    semilogx= kwargs.pop('semilogx',False)
    semilogy= kwargs.pop('semilogy',False)
    colorbar= kwargs.pop('colorbar',False)
    onedhisttype= kwargs.pop('onedhisttype','step')
    onedhistcolor= kwargs.pop('onedhistcolor','k')
    onedhistfc= kwargs.pop('onedhistfc','w')
    onedhistec= kwargs.pop('onedhistec','k')
    onedhistxnormed= kwargs.pop('onedhistxnormed',True)
    onedhistynormed= kwargs.pop('onedhistynormed',True)
    onedhistxweights= kwargs.pop('onedhistxweights',None)
    onedhistyweights= kwargs.pop('onedhistyweights',None)
    if 'bins' in kwargs:
        bins= kwargs['bins']
        kwargs.pop('bins')
    elif onedhists:
        if isinstance(args[0],sc.ndarray):
            bins= round(0.3*sc.sqrt(args[0].shape[0]))
        elif isinstance(args[0],list):
            bins= round(0.3*sc.sqrt(len(args[0])))
        else:
            bins= 30
    if onedhists:
        if overplot or gcf: fig= pyplot.gcf()
        else: fig= pyplot.figure()
        nullfmt   = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width
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
    elif not overplot and not gcf: pyplot.figure()
    ax=pyplot.gca()
    ax.set_autoscale_on(False)
    xlabel= kwargs.pop('xlabel',None)
    ylabel= kwargs.pop('ylabel',None)
    clabel= kwargs.pop('clabel',None)
    xlimits= kwargs.pop('xrange',None)
    if xlimits is None:
        if isinstance(args[0],list):
            xlimits=(sc.array(args[0]).min(),sc.array(args[0]).max())
        else:
            xlimits=(args[0].min(),args[0].max())
    ylimits= kwargs.pop('yrange',None)
    if ylimits is None:
        if isinstance(args[1],list):
            ylimits=(sc.array(args[1]).min(),sc.array(args[1]).max())
        else:
            ylimits=(args[1].min(),args[1].max())
    climits= kwargs.pop('crange',None)
    if climits is None and scatter:
        if 'c' in kwargs and isinstance(kwargs['c'],list):
            climits=(sc.array(kwargs['c']).min(),sc.array(kwargs['c']).max())
        elif 'c' in kwargs:
            climits=(kwargs['c'].min(),kwargs['c'].max())
        else:
            climits= None
    if scatter:
        out= pyplot.scatter(*args,**kwargs)
    elif loglog:
        out= pyplot.loglog(*args,**kwargs)
    elif semilogx:
        out= pyplot.semilogx(*args,**kwargs)
    elif semilogy:
        out= pyplot.semilogy(*args,**kwargs)
    else:
        out= pyplot.plot(*args,**kwargs)
    if overplot:
        pass
    else:
        if semilogy:
            ax= pyplot.gca()
            ax.set_yscale('log')
        elif semilogx:
            ax= pyplot.gca()
            ax.set_xscale('log')
        elif loglog:
            ax= pyplot.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')
        pyplot.xlim(*xlimits)
        pyplot.ylim(*ylimits)
        _add_axislabels(xlabel,ylabel)
        if not semilogy and not semilogx and not loglog:
            _add_ticks()
        elif semilogy:
            _add_ticks(xticks=True,yticks=False)
        elif semilogx:
            _add_ticks(yticks=True,xticks=False)
    #Add colorbar
    if colorbar:
        cbar= pyplot.colorbar(out,fraction=0.15)
        cbar.set_clim(*climits)
        if not clabel is None:
            cbar.set_label(clabel)
    #Add onedhists
    if not onedhists:
        return out
    histx, edges, patches= axHistx.hist(args[0], bins=bins,
                                        normed=onedhistxnormed,
                                        weights=onedhistxweights,
                                        histtype=onedhisttype,
                                        range=sorted(xlimits),
                                        color=onedhistcolor,fc=onedhistfc,
                                        ec=onedhistec)
    histy, edges, patches= axHisty.hist(args[1], bins=bins,
                                        orientation='horizontal',
                                        weights=onedhistyweights,
                                        normed=onedhistynormed,
                                        histtype=onedhisttype,
                                        range=sorted(ylimits),
                                        color=onedhistcolor,fc=onedhistfc,
                                        ec=onedhistec)
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    axHistx.set_ylim( 0, 1.2*sc.amax(histx))
    axHisty.set_xlim( 0, 1.2*sc.amax(histy))
    return (axScatter,axHistx,axHisty)

def bovy_plot3d(*args,**kwargs):
    """
    NAME:

       bovy_plot3d

    PURPOSE:

       plot in 3d much as in 2d

    INPUT:

       see http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       xrange

       yrange

       overplot=True does not start a new figure

    OUTPUT:

    HISTORY:

       2011-01-08 - Written - Bovy (NYU)

    """
    overplot= kwargs.pop('overplot',False)
    if not overplot: pyplot.figure()
    ax=pyplot.gca(projection='3d')
    ax.set_autoscale_on(False)
    xlabel= kwargs.pop('xlabel',None)
    ylabel= kwargs.pop('ylabel',None)
    zlabel= kwargs.pop('zlabel',None)
    if 'xrange' in kwargs:
        xlimits= kwargs.pop('xrange')
    else:
        if isinstance(args[0],list):
            xlimits=(sc.array(args[0]).min(),sc.array(args[0]).max())
        else:
            xlimits=(args[0].min(),args[0].max())
    if 'yrange' in kwargs:
        ylimits= kwargs.pop('yrange')
    else:
        if isinstance(args[1],list):
            ylimits=(sc.array(args[1]).min(),sc.array(args[1]).max())
        else:
            ylimits=(args[1].min(),args[1].max())
    if 'zrange' in kwargs:
        zlimits= kwargs.pop('zrange')
    else:
        if isinstance(args[2],list):
            zlimits=(sc.array(args[2]).min(),sc.array(args[2]).max())
        else:
            zlimits=(args[1].min(),args[2].max())
    out= pyplot.plot(*args,**kwargs)
    if overplot:
        pass
    else:
        if xlabel != None:
            if xlabel[0] != '$':
                thisxlabel=r'$'+xlabel+'$'
            else:
                thisxlabel=xlabel
            ax.set_xlabel(thisxlabel)
        if ylabel != None:
            if ylabel[0] != '$':
                thisylabel=r'$'+ylabel+'$'
            else:
                thisylabel=ylabel
            ax.set_ylabel(thisylabel)
        if zlabel != None:
            if zlabel[0] != '$':
                thiszlabel=r'$'+zlabel+'$'
            else:
                thiszlabel=zlabel
            ax.set_zlabel(thiszlabel)
        ax.set_xlim3d(*xlimits)
        ax.set_ylim3d(*ylimits)
        ax.set_zlim3d(*zlimits)
    return out

def bovy_dens2d(X,**kwargs):
    """
    NAME:

       bovy_dens2d

    PURPOSE:

       plot a 2d density with optional contours

    INPUT:

       first argument is the density

       matplotlib.pyplot.imshow keywords (see http://matplotlib.sourceforge.net/api/axes_api.html#matplotlib.axes.Axes.imshow)

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       xrange

       yrange

       noaxes - don't plot any axes

       overplot - if True, overplot

       colorbar - if True, add colorbar

       shrink= colorbar argument: shrink the colorbar by the factor (optional)

       conditional - normalize each column separately (for probability densities, i.e., cntrmass=True)

       gcf=True does not start a new figure (does change the ranges and labels)

       Contours:
       
       justcontours - if True, only draw contours

       contours - if True, draw contours (10 by default)

       levels - contour-levels

       cntrmass - if True, the density is a probability and the levels are probability masses contained within the contour

       cntrcolors - colors for contours (single color or array)

       cntrlabel - label the contours

       cntrlw, cntrls - linewidths and linestyles for contour

       cntrlabelsize, cntrlabelcolors,cntrinline - contour arguments

       cntrSmooth - use ndimage.gaussian_filter to smooth before contouring

       onedhists - if True, make one-d histograms on the sides

       onedhistcolor - histogram color

       retAxes= return all Axes instances

       retCont= return the contour instance

    OUTPUT:

       plot to output device, Axes instances depending on input

    HISTORY:

       2010-03-09 - Written - Bovy (NYU)

    """
    overplot= kwargs.pop('overplot',False)
    gcf= kwargs.pop('gcf',False)
    if not overplot and not gcf:
        pyplot.figure()
    xlabel= kwargs.pop('xlabel',None)
    ylabel= kwargs.pop('ylabel',None)
    zlabel= kwargs.pop('zlabel',None)
    if 'extent' in kwargs:
        extent= kwargs.pop('extent')
    else:
        xlimits= kwargs.pop('xrange',[0,X.shape[1]])
        ylimits= kwargs.pop('yrange',[0,X.shape[0]])
        extent= xlimits+ylimits
    if not 'aspect' in kwargs:
        kwargs['aspect']= (xlimits[1]-xlimits[0])/float(ylimits[1]-ylimits[0])
    noaxes= kwargs.pop('noaxes',False)
    justcontours= kwargs.pop('justcontours',False)
    if ('contours' in kwargs and kwargs['contours']) or \
            'levels' in kwargs or justcontours or \
            ('cntrmass' in kwargs and kwargs['cntrmass']):
        contours= True
    else:
        contours= False
    kwargs.pop('contours',None)
    if 'levels' in kwargs:
        levels= kwargs['levels']
        kwargs.pop('levels')
    elif contours:
        if 'cntrmass' in kwargs and kwargs['cntrmass']:
            levels= sc.linspace(0.,1.,_DEFAULTNCNTR)
        elif True in sc.isnan(sc.array(X)):
            levels= sc.linspace(sc.nanmin(X),sc.nanmax(X),_DEFAULTNCNTR)
        else:
            levels= sc.linspace(sc.amin(X),sc.amax(X),_DEFAULTNCNTR)
    cntrmass= kwargs.pop('cntrmass',False)
    conditional= kwargs.pop('conditional',False)
    cntrcolors= kwargs.pop('cntrcolors','k')
    cntrlabel= kwargs.pop('cntrlabel',False)
    cntrlw= kwargs.pop('cntrlw',None)
    cntrls= kwargs.pop('cntrls',None)
    cntrSmooth= kwargs.pop('cntrSmooth',None)
    cntrlabelsize= kwargs.pop('cntrlabelsize',None)
    cntrlabelcolors= kwargs.pop('cntrlabelcolors',None)
    cntrinline= kwargs.pop('cntrinline',None)
    retCumImage= kwargs.pop('retCumImage',False)
    cb= kwargs.pop('colorbar',False)
    shrink= kwargs.pop('shrink',None)
    onedhists= kwargs.pop('onedhists',False)
    onedhistcolor= kwargs.pop('onedhistcolor','k')
    retAxes= kwargs.pop('retAxes',False)
    retCont= kwargs.pop('retCont',False)
    if onedhists:
        if overplot or gcf: fig= pyplot.gcf()
        else: fig= pyplot.figure()
        nullfmt   = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width
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
    ax=pyplot.gca()
    ax.set_autoscale_on(False)
    if conditional:
        plotthis= X/sc.tile(sc.sum(X,axis=0),(X.shape[1],1))
    else:
        plotthis= X
    if not justcontours:
        out= pyplot.imshow(plotthis,extent=extent,**kwargs)
    if not overplot:
        pyplot.axis(extent)
        _add_axislabels(xlabel,ylabel)
        _add_ticks()
    #Add colorbar
    if cb and not justcontours:
        if shrink is None:
            shrink= sc.amin([float(kwargs.pop('aspect',1.))*0.87,1.])
        CB1= pyplot.colorbar(out,shrink=shrink)
        if not zlabel is None:
            if zlabel[0] != '$':
                thiszlabel=r'$'+zlabel+'$'
            else:
                thiszlabel=zlabel
            CB1.set_label(thiszlabel)
    if contours or retCumImage:
        aspect= kwargs.get('aspect',None)
        origin= kwargs.get('origin',None)
        if cntrmass:
            #Sum from the top down!
            plotthis[sc.isnan(plotthis)]= 0.
            sortindx= sc.argsort(plotthis.flatten())[::-1]
            cumul= sc.cumsum(sc.sort(plotthis.flatten())[::-1])/sc.sum(plotthis.flatten())
            cntrThis= sc.zeros(sc.prod(plotthis.shape))
            cntrThis[sortindx]= cumul
            cntrThis= sc.reshape(cntrThis,plotthis.shape)
        else:
            cntrThis= plotthis
        if contours:
            if not cntrSmooth is None:
                cntrThis= ndimage.gaussian_filter(cntrThis,cntrSmooth,
                                                  mode='nearest')
            cont= pyplot.contour(cntrThis,levels,colors=cntrcolors,
                                 linewidths=cntrlw,extent=extent,aspect=aspect,
                                 linestyles=cntrls,origin=origin)
            if cntrlabel:
                pyplot.clabel(cont,fontsize=cntrlabelsize,
                              colors=cntrlabelcolors,
                              inline=cntrinline)
    if noaxes:
        ax.set_axis_off()
    #Add onedhists
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
    histx= sc.nansum(X.T,axis=1)*m.fabs(ylimits[1]-ylimits[0])/X.shape[1] #nansum bc nan is *no dens value*
    histy= sc.nansum(X.T,axis=0)*m.fabs(xlimits[1]-xlimits[0])/X.shape[0]
    histx[sc.isnan(histx)]= 0.
    histy[sc.isnan(histy)]= 0.
    dx= (extent[1]-extent[0])/float(len(histx))
    axHistx.plot(sc.linspace(extent[0]+dx,extent[1]-dx,len(histx)),histx,
                 drawstyle='steps-mid',color=onedhistcolor)
    dy= (extent[3]-extent[2])/float(len(histy))
    axHisty.plot(histy,sc.linspace(extent[2]+dy,extent[3]-dy,len(histy)),
                 drawstyle='steps-mid',color=onedhistcolor)
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    axHistx.set_ylim( 0, 1.2*sc.amax(histx))
    axHisty.set_xlim( 0, 1.2*sc.amax(histy))
    if retCumImage:
        return cntrThis
    elif retAxes:
        return (axScatter,axHistx,axHisty)
    elif justcontours:
        return cntrThis
    else:
        return out

def bovy_print(fig_width=5,fig_height=5,axes_labelsize=16,
               text_fontsize=11,legend_fontsize=12,
               xtick_labelsize=10,ytick_labelsize=10,
               xtick_minor_size=2,ytick_minor_size=2,
               xtick_major_size=4,ytick_major_size=4):
    """
    NAME:

       bovy_print

    PURPOSE:

       setup a figure for plotting

    INPUT:

       fig_width - width in inches

       fig_height - height in inches

       axes_labelsize - size of the axis-labels

       text_fontsize - font-size of the text (if any)

       legend_fontsize - font-size of the legend (if any)

       xtick_labelsize - size of the x-axis labels

       ytick_labelsize - size of the y-axis labels

       xtick_minor_size - size of the minor x-ticks

       ytick_minor_size - size of the minor y-ticks

    OUTPUT:

       (none)

    HISTORY:

       2009-12-23 - Written - Bovy (NYU)

    """
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': axes_labelsize,
              'font.size': text_fontsize,
              'legend.fontsize': legend_fontsize,
              'xtick.labelsize':xtick_labelsize,
              'ytick.labelsize':ytick_labelsize,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : xtick_major_size,
              'ytick.major.size' : ytick_major_size,
              'xtick.minor.size' : xtick_minor_size,
              'ytick.minor.size' : ytick_minor_size,
              'legend.numpoints':1,
              'xtick.top': True,
              'xtick.direction': 'in',
              'ytick.right': True,
              'ytick.direction': 'in'}
    pyplot.rcParams.update(params)
    rc('text.latex', preamble=r'\usepackage{amsmath}'+'\n'
       +r'\usepackage{amssymb}')

def bovy_text(*args,**kwargs):
    """
    NAME:

       bovy_text

    PURPOSE:

       thin wrapper around matplotlib's text and annotate

       use keywords:

          'bottom_left=True'

          'bottom_right=True'

          'top_left=True'

          'top_right=True'

          'title=True'

       to place the text in one of the corners or use it as the title

    INPUT:

       see matplotlib's text
          (http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.text)

    OUTPUT:

       prints text on the current figure

    HISTORY:

       2010-01-26 - Written - Bovy (NYU)

    """
    if kwargs.pop('title',False):
        pyplot.annotate(args[0],(0.5,1.05),xycoords='axes fraction',
                        horizontalalignment='center',
                        verticalalignment='top',**kwargs)
    elif kwargs.pop('bottom_left',False):
        pyplot.annotate(args[0],(0.05,0.05),xycoords='axes fraction',**kwargs)
    elif kwargs.pop('bottom_right',False):
        pyplot.annotate(args[0],(0.95,0.05),xycoords='axes fraction',
                        horizontalalignment='right',**kwargs)
    elif kwargs.pop('top_right',False):
        pyplot.annotate(args[0],(0.95,0.95),xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='top',**kwargs)
    elif kwargs.pop('top_left',False):
        pyplot.annotate(args[0],(0.05,0.95),xycoords='axes fraction',
                        verticalalignment='top',**kwargs)
    else:
        pyplot.text(*args,**kwargs)

def scatterplot(x,y,*args,**kwargs):
    """
    NAME:

       scatterplot

    PURPOSE:

       make a 'smart' scatterplot that is a density plot in high-density
       regions and a regular scatterplot for outliers

    INPUT:

       x, y

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       xrange

       yrange

       bins - number of bins to use in each dimension

       weights - data-weights

       aspect - aspect ratio

       conditional - normalize each column separately (for probability densities, i.e., cntrmass=True)

       gcf=True does not start a new figure (does change the ranges and labels)

       contours - if False, don't plot contours

       justcontours - if True, only draw contours, no density

       cntrcolors - color of contours (can be array as for bovy_dens2d)

       cntrlw, cntrls - linewidths and linestyles for contour

       cntrSmooth - use ndimage.gaussian_filter to smooth before contouring

       levels - contour-levels; data points outside of the last level will be individually shown (so, e.g., if this list is descending, contours and data points will be overplotted)

       onedhists - if True, make one-d histograms on the sides

       onedhistx - if True, make one-d histograms on the side of the x distribution

       onedhisty - if True, make one-d histograms on the side of the y distribution

       onedhistcolor, onedhistfc, onedhistec

       onedhistxnormed, onedhistynormed - normed keyword for one-d histograms
       
       onedhistxweights, onedhistyweights - weights keyword for one-d histograms

       cmap= cmap for density plot

       hist= and edges= - you can supply the histogram of the data yourself, this can be useful if you want to censor the data, both need to be set and calculated using scipy.histogramdd with the given range

       retAxes= return all Axes instances

    OUTPUT:

       plot to output device, Axes instance(s) or not, depending on input

    HISTORY:

       2010-04-15 - Written - Bovy (NYU)

    """
    xlabel= kwargs.pop('xlabel',None)
    ylabel= kwargs.pop('ylabel',None)
    if 'xrange' in kwargs:
        xrange= kwargs.pop('xrange')
    else:
        if isinstance(x,list): xrange=[sc.amin(x),sc.amax(x)]
        else: xrange=[x.min(),x.max()]
    if 'yrange' in kwargs:
        yrange= kwargs.pop('yrange')
    else:
        if isinstance(y,list): yrange=[sc.amin(y),sc.amax(y)]
        else: yrange=[y.min(),y.max()]
    ndata= len(x)
    bins= kwargs.pop('bins',round(0.3*sc.sqrt(ndata)))
    weights= kwargs.pop('weights',None)
    levels= kwargs.pop('levels',special.erf(sc.arange(1,4)/sc.sqrt(2.)))
    aspect= kwargs.pop('aspect',(xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    conditional= kwargs.pop('conditional',False)
    contours= kwargs.pop('contours',True)
    justcontours= kwargs.pop('justcontours',False)
    cntrcolors= kwargs.pop('cntrcolors','k')
    cntrlw= kwargs.pop('cntrlw',None)
    cntrls= kwargs.pop('cntrls',None)
    cntrSmooth= kwargs.pop('cntrSmooth',None)
    onedhists= kwargs.pop('onedhists',False)
    onedhistx= kwargs.pop('onedhistx',onedhists)
    onedhisty= kwargs.pop('onedhisty',onedhists)
    onedhisttype= kwargs.pop('onedhisttype','step')
    onedhistcolor= kwargs.pop('onedhistcolor','k')
    onedhistfc= kwargs.pop('onedhistfc','w')
    onedhistec= kwargs.pop('onedhistec','k')
    onedhistls= kwargs.pop('onedhistls','solid')
    onedhistlw= kwargs.pop('onedhistlw',None)
    onedhistsbins= kwargs.pop('onedhistsbins',round(0.3*sc.sqrt(ndata)))
    overplot= kwargs.pop('overplot',False)
    gcf= kwargs.pop('gcf',False)
    cmap= kwargs.pop('cmap',cm.gist_yarg)
    onedhistxnormed= kwargs.pop('onedhistxnormed',True)
    onedhistynormed= kwargs.pop('onedhistynormed',True)
    onedhistxweights= kwargs.pop('onedhistxweights',weights)
    onedhistyweights= kwargs.pop('onedhistyweights',weights)
    retAxes= kwargs.pop('retAxes',False)
    if onedhists or onedhistx or onedhisty:
        if overplot or gcf: fig= pyplot.gcf()
        else: fig= pyplot.figure()
        nullfmt   = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width
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
    data= sc.array([x,y]).T
    if 'hist' in kwargs and 'edges' in kwargs:
        hist=kwargs['hist']
        kwargs.pop('hist')
        edges=kwargs['edges']
        kwargs.pop('edges')
    else:
        hist, edges= sc.histogramdd(data,bins=bins,range=[xrange,yrange],
                                    weights=weights)
    if contours:
        cumimage= bovy_dens2d(hist.T,contours=contours,levels=levels,
                              cntrmass=contours,cntrSmooth=cntrSmooth,
                              cntrcolors=cntrcolors,cmap=cmap,origin='lower',
                              xrange=xrange,yrange=yrange,xlabel=xlabel,
                              ylabel=ylabel,interpolation='nearest',
                              retCumImage=True,aspect=aspect,
                              conditional=conditional,
                              cntrlw=cntrlw,cntrls=cntrls,
                              justcontours=justcontours,zorder=5*justcontours,
                              overplot=(gcf or onedhists or overplot or onedhistx or onedhisty))
    else:
        cumimage= bovy_dens2d(hist.T,contours=contours,
                              cntrcolors=cntrcolors,
                              cmap=cmap,origin='lower',
                              xrange=xrange,yrange=yrange,xlabel=xlabel,
                              ylabel=ylabel,interpolation='nearest',
                              conditional=conditional,
                              retCumImage=True,aspect=aspect,
                              cntrlw=cntrlw,cntrls=cntrls,
                              overplot=(gcf or onedhists or overplot or onedhistx or onedhisty))
    #Set axes and labels
    pyplot.axis(list(xrange)+list(yrange))
    if not overplot:
        _add_axislabels(xlabel,ylabel)
        _add_ticks()
    binxs= []
    xedge= edges[0]
    for ii in range(len(xedge)-1):
        binxs.append((xedge[ii]+xedge[ii+1])/2.)
    binxs= sc.array(binxs)
    binys= []
    yedge= edges[1]
    for ii in range(len(yedge)-1):
        binys.append((yedge[ii]+yedge[ii+1])/2.)
    binys= sc.array(binys)
    cumInterp= interpolate.RectBivariateSpline(binxs,binys,cumimage.T,
                                               kx=1,ky=1)
    cums= []
    for ii in range(len(x)):
        cums.append(cumInterp(x[ii],y[ii])[0,0])
    cums= sc.array(cums)
    plotx= x[cums > levels[-1]]
    ploty= y[cums > levels[-1]]
    if not len(plotx) == 0:
        if not weights == None:
            w8= weights[cums > levels[-1]]
            for ii in range(len(plotx)):
                bovy_plot(plotx[ii],ploty[ii],overplot=True,
                          color='%.2f'%(1.-w8[ii]),*args,**kwargs)
        else:
            bovy_plot(plotx,ploty,overplot=True,zorder=1,*args,**kwargs)
    #Add onedhists
    if not (onedhists or onedhistx or onedhisty):
        if retAxes:
            return pyplot.gca()
        else:
            return None
    if onedhistx:
        histx, edges, patches= axHistx.hist(x,bins=onedhistsbins,
                                            normed=onedhistxnormed,
                                            weights=onedhistxweights,
                                            histtype=onedhisttype,
                                            range=sorted(xrange),
                                            color=onedhistcolor,fc=onedhistfc,
                                            ec=onedhistec,ls=onedhistls,
                                            lw=onedhistlw)
    if onedhisty:
        histy, edges, patches= axHisty.hist(y,bins=onedhistsbins,
                                            orientation='horizontal',
                                            weights=onedhistyweights,
                                            normed=onedhistynormed,
                                            histtype=onedhisttype,
                                            range=sorted(yrange),
                                            color=onedhistcolor,fc=onedhistfc,
                                            ec=onedhistec,ls=onedhistls,
                                            lw=onedhistlw)
    if onedhistx and not overplot:
        axHistx.set_xlim( axScatter.get_xlim() )
        axHistx.set_ylim( 0, 1.2*sc.amax(histx))
    if onedhisty and not overplot:
        axHisty.set_ylim( axScatter.get_ylim() )
        axHisty.set_xlim( 0, 1.2*sc.amax(histy))
    if not onedhistx: axHistx= None
    if not onedhisty: axHisty= None
    if retAxes:
        return (axScatter,axHistx,axHisty)
    else:
        return None

def _add_axislabels(xlabel,ylabel):
    """
    NAME:

       _add_axislabels

    PURPOSE:

       add axis labels to the current figure

    INPUT:

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

    OUTPUT:

       (none; works on the current axes)

    HISTORY:

       2009-12-23 - Written - Bovy (NYU)

    """
    if xlabel != None:
        if xlabel[0] != '$':
            thisxlabel=r'$'+xlabel+'$'
        else:
            thisxlabel=xlabel
        pyplot.xlabel(thisxlabel)
    if ylabel != None:
        if ylabel[0] != '$':
            thisylabel=r'$'+ylabel+'$'
        else:
            thisylabel=ylabel
        pyplot.ylabel(thisylabel)
        
def _add_ticks(xticks=True,yticks=True):
    """
    NAME:

       _add_ticks

    PURPOSE:

       add minor axis ticks to a plot

    INPUT:

       (none; works on the current axes)

    OUTPUT:

       (none; works on the current axes)

    HISTORY:

       2009-12-23 - Written - Bovy (NYU)

    """
    ax=pyplot.gca()
    if xticks:
        xstep= ax.xaxis.get_majorticklocs()
        xstep= xstep[1]-xstep[0]
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xstep/5.))
    if yticks:
        ystep= ax.yaxis.get_majorticklocs()
        ystep= ystep[1]-ystep[0]
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ystep/5.))


class GalPolarAxes(PolarAxes):
    '''
    A variant of PolarAxes where theta increases clockwise
    '''
    name = 'galpolar'

    class GalPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy   = sc.zeros(tr.shape, sc.float_)
            t    = tr[:, 0:1]
            r    = tr[:, 1:2]
            x    = xy[:, 0:1]
            y    = xy[:, 1:2]
            x[:] = r * sc.cos(t)
            y[:] = -r * sc.sin(t)
            return xy

        transform_non_affine = transform

        def inverted(self):
            return GalPolarAxes.InvertedGalPolarTransform()

    class InvertedGalPolarTransform(PolarAxes.InvertedPolarTransform):
        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = sc.sqrt(x*x + y*y)
            theta = sc.arctan2(y, x)
            return sc.concatenate((theta, r), 1)

        def inverted(self):
            return GalPolarAxes.GalPolarTransform()

    def _set_lim_and_transforms(self):
        PolarAxes._set_lim_and_transforms(self)
        self.transProjection = self.GalPolarTransform()
        self.transData = (
            self.transScale + 
            self.transProjection + 
            (self.transProjectionAffine + self.transAxes))
        self._xaxis_transform = (
            self.transProjection +
            self.PolarAffine(IdentityTransform(), Bbox.unit()) +
            self.transAxes)
        self._xaxis_text1_transform = (
            self._theta_label1_position +
            self._xaxis_transform)
        self._yaxis_transform = (
            Affine2D().scale(sc.pi * 2.0, 1.0) +
            self.transData)
        self._yaxis_text1_transform = (
            self._r_label1_position +
            Affine2D().scale(1.0 / 360.0, 1.0) +
            self._yaxis_transform)

register_projection(GalPolarAxes)    
