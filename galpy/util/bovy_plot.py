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
#############################################################################
#############################################################################
#Copyright (c) 2010, Jo Bovy
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
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import NullFormatter
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
    if kwargs.has_key('format'):
        pyplot.savefig(filename,format=kwags['format'])
    else:
        pyplot.savefig(filename,format=re.split(r'\.',filename)[-1])
    
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
       + all pyplot.hist keywords
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
    out= pyplot.hist(x,**kwargs)
    _add_axislabels(xlabel,ylabel)
    if not kwargs.has_key('range'):
        pyplot.xlim(x.min(),x.max())
    else:
        pyplot.xlim(kwargs['range'])
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
       overplot=True does not start a new figure
    OUTPUT:
    HISTORY:
       2009-12-28 - Written - Bovy (NYU)
    """
    if kwargs.has_key('overplot') and kwargs['overplot']:
        kwargs.pop('overplot')
        overplot=True
    elif kwargs.has_key('overplot'):
        kwargs.pop('overplot')
        pyplot.figure()
        overplot=False
    else:
        pyplot.figure()
        overplot=False
    ax=pyplot.gca()
    ax.set_autoscale_on(False)
    if kwargs.has_key('xlabel'):
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')
    else:
        xlabel=None
    if kwargs.has_key('ylabel'):
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')
    else:
        ylabel=None
    if kwargs.has_key('xrange'):
        xlimits=kwargs['xrange']
        kwargs.pop('xrange')
    else:
        xlimits=(args[0].min(),args[0].max())
    if kwargs.has_key('yrange'):
        ylimits=kwargs['yrange']
        kwargs.pop('yrange')
    else:
        ylimits=(args[1].min(),args[1].max())
    out= pyplot.plot(*args,**kwargs)
    if overplot:
        pass
    else:
        pyplot.xlim(*xlimits)
        pyplot.ylim(*ylimits)
        _add_axislabels(xlabel,ylabel)
        _add_ticks()
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
       
       Contours:
       
       contours - if True, draw contours (10 by default)
       levels - contour-levels
       cntrmass - if True, the density is a probability and the levels 
                  are probability masses contained within the contour
       cntrcolors - colors for contours (single color or array)
       cntrlabel - label the contours
       cntrlw, cntrls - linewidths and linestyles for contour
       cntrlabelsize, cntrlabelcolors,cntrinline - contour arguments
    OUTPUT:
    HISTORY:
       2010-03-09 - Written - Bovy (NYU)
    """
    if kwargs.has_key('overplot'):
        overplot= kwargs['overplot']
        kwargs.pop('overplot')
    else:
        overplot= False
    if not overplot:
        pyplot.figure()
    ax=pyplot.gca()
    ax.set_autoscale_on(False)
    if kwargs.has_key('xlabel'):
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')
    else:
        xlabel=None
    if kwargs.has_key('ylabel'):
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')
    else:
        ylabel=None
    if kwargs.has_key('extent'):
        extent= kwargs['extent']
        kwargs.pop('extent')
    else:
        if kwargs.has_key('xrange'):
            xlimits=list(kwargs['xrange'])
            kwargs.pop('xrange')
        else:
            xlimits=[0,X.shape[0]]
        if kwargs.has_key('yrange'):
            ylimits=list(kwargs['yrange'])
            kwargs.pop('yrange')
        else:
            ylimits=[0,X.shape[1]]
        extent= xlimits+ylimits
    if kwargs.has_key('noaxes'):
        noaxes= kwargs['noaxes']
        kwargs.pop('noaxes')
    else:
        noaxes= False
    if kwargs.has_key('contours') and kwargs['contours']:
        contours= True
        kwargs.pop('contours')
    elif kwargs.has_key('levels') or kwargs.has_key('cntrmass'):
        contours= True
    else:
        contours= False
        if kwargs.has_key('contours'): kwargs.pop('contours')
    if kwargs.has_key('levels'):
        levels= kwargs['levels']
        kwargs.pop('levels')
    elif contours:
        if kwargs.has_key('cntrmass') and kwargs['cntrmass']:
            levels= sc.linspace(0.,1.,_DEFAULTNCNTR)
        elif True in sc.isnan(sc.array(X)):
            levels= sc.linspace(sc.nanmin(X),sc.nanmax(X),_DEFAULTNCNTR)
        else:
            levels= sc.linspace(sc.amin(X),sc.amax(X),_DEFAULTNCNTR)
    if kwargs.has_key('cntrmass') and kwargs['cntrmass']:
        cntrmass= True
        kwargs.pop('cntrmass')
    else:
        cntrmass= False
        if kwargs.has_key('cntrmass'): kwargs.pop('cntrmass')
    if kwargs.has_key('cntrcolors'):
        cntrcolors= kwargs['cntrcolors']
        kwargs.pop('cntrcolors')
    elif contours:
        cntrcolors='k'
    if kwargs.has_key('cntrlabel') and kwargs['cntrlabel']:
        cntrlabel= True
        kwargs.pop('cntrlabel')
    else:
        cntrlabel= False
        if kwargs.has_key('cntrlabel'): kwargs.pop('cntrlabel')
    if kwargs.has_key('cntrlw'):
        cntrlw= kwargs['cntrlw']
        kwargs.pop('cntrlw')
    elif contours:
        cntrlw= None
    if kwargs.has_key('cntrls'):
        cntrls= kwargs['cntrls']
        kwargs.pop('cntrls')
    elif contours:
        cntrls= None
    if kwargs.has_key('cntrlabelsize'):
        cntrlabelsize= kwargs['cntrlabelsize']
        kwargs.pop('cntrlabelsize')
    elif contours:
        cntrlabelsize= None
    if kwargs.has_key('cntrlabelcolors'):
        cntrlabelcolors= kwargs['cntrlabelcolors']
        kwargs.pop('cntrlabelcolors')
    elif contours:
        cntrlabelcolors= None
    if kwargs.has_key('cntrinline'):
        cntrinline= kwargs['cntrinline']
        kwargs.pop('cntrinline')
    elif contours:
        cntrinline= None
    if kwargs.has_key('retCumImage'):
        retCumImage= kwargs['retCumImage']
        kwargs.pop('retCumImage')
    else:
        retCumImage= False
    out= pyplot.imshow(X,extent=extent,**kwargs)
    pyplot.axis(extent)
    _add_axislabels(xlabel,ylabel)
    _add_ticks()
    if contours:
        if kwargs.has_key('aspect'):
            aspect= kwargs['aspect']
        else:
            aspect= None
        if kwargs.has_key('origin'):
            origin= kwargs['origin']
        else:
            origin= None
        if cntrmass:
            #Sum from the top down!
            sortindx= sc.argsort(X.flatten())[::-1]
            cumul= sc.cumsum(sc.sort(X.flatten())[::-1])/sc.sum(X.flatten())
            cntrThis= sc.zeros(sc.prod(X.shape))
            cntrThis[sortindx]= cumul
            cntrThis= sc.reshape(cntrThis,X.shape)
        else:
            cntrThis= X
        cont= pyplot.contour(cntrThis,levels,colors=cntrcolors,
                             linewidths=cntrlw,extent=extent,aspect=aspect,
                             linestyles=cntrls,origin=origin)
        if cntrlabel:
            pyplot.clabel(cont,fontsize=cntrlabelsize,colors=cntrlabelcolors,
                          inline=cntrinline)
    if noaxes:
        ax.set_axis_off()
    if retCumImage:
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
              'text.fontsize': text_fontsize,
              'legend.fontsize': legend_fontsize,
              'xtick.labelsize':xtick_labelsize,
              'ytick.labelsize':ytick_labelsize,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : xtick_major_size,
              'ytick.major.size' : ytick_major_size,
              'xtick.minor.size' : xtick_minor_size,
              'ytick.minor.size' : ytick_minor_size}
    pyplot.rcParams.update(params)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

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
    if kwargs.has_key('title'):
        kwargs.pop('title')
        pyplot.annotate(args[0],(0.5,1.05),xycoords='axes fraction',
                        horizontalalignment='center',
                        verticalalignment='top')
    elif kwargs.has_key('bottom_left'):
        kwargs.pop('bottom_left')
        pyplot.annotate(args[0],(0.05,0.05),xycoords='axes fraction')
    elif kwargs.has_key('bottom_right'):
        kwargs.pop('bottom_right')
        pyplot.annotate(args[0],(0.95,0.05),xycoords='axes fraction',
                        horizontalalignment='right')
    elif kwargs.has_key('top_right'):
        kwargs.pop('top_right')
        pyplot.annotate(args[0],(0.95,0.95),xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='top')
    elif kwargs.has_key('top_left'):
        kwargs.pop('top_left')
        pyplot.annotate(args[0],(0.05,0.95),xycoords='axes fraction',
                        verticalalignment='top')
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
       onedhists - if True, make one-d histograms on the sides
       onedhistcolor, onedhistfc, onedhistec
    OUTPUT:
    HISTORY:
       2010-04-15 - Written - Bovy (NYU)
    """
    if kwargs.has_key('xlabel'):
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')
    else:
        xlabel=None
    if kwargs.has_key('ylabel'):
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')
    else:
        ylabel=None
    if kwargs.has_key('xrange'):
        xrange=kwargs['xrange']
        kwargs.pop('xrange')
    else:
        xrange=[x.min(),x.max()]
    if kwargs.has_key('yrange'):
        yrange=kwargs['yrange']
        kwargs.pop('yrange')
    else:
        yrange=[y.min(),y.max()]
    ndata= len(x)
    if kwargs.has_key('bins'):
        bins= kwargs['bins']
        kwargs.pop('bins')
    else:
        bins= round(0.3*sc.sqrt(ndata))
    if kwargs.has_key('weights'):
        weights= kwargs['weights']
        kwargs.pop('weights')
    else:
        weights= None
    if kwargs.has_key('levels'):
        levels= kwargs['levels']
        kwargs.pop('levels')
    else:
        levels= special.erf(0.5*sc.arange(1,4))
    if kwargs.has_key('aspect'):
        aspect= kwargs['aspect']
        kwargs.pop('aspect')
    else:
        aspect= (xrange[1]-xrange[0])/(yrange[1]-yrange[0])
    if kwargs.has_key('onedhists'):
        onedhists= kwargs['onedhists']
        kwargs.pop('onedhists')
    else:
        onedhists= False
    if kwargs.has_key('onedhisttype'):
        onedhisttype= kwargs['onedhisttype']
        kwargs.pop('onedhisttype')
    else:
        onedhisttype= 'step'
    if kwargs.has_key('onedhistcolor'):
        onedhistcolor= kwargs['onedhistcolor']
        kwargs.pop('onedhistcolor')
    else:
        onedhistcolor= 'k'
    if kwargs.has_key('onedhistfc'):
        onedhistfc=kwargs['onedhistfc']
        kwargs.pop('onedhistfc')
    else:
        onedhistfc= 'w'
    if kwargs.has_key('onedhistec'):
        onedhistec=kwargs['onedhistec']
        kwargs.pop('onedhistec')
    else:
        onedhistec= 'k'
    if onedhists:
        fig= pyplot.figure()
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
    data= sc.array([x,y]).T
    hist, edges= sc.histogramdd(data,bins=bins,range=[xrange,yrange],
                                weights=weights)
    cumimage= bovy_dens2d(hist.T,contours=True,levels=levels,cntrmass=True,
                          cntrcolors='k',cmap=cm.gist_yarg,origin='lower',
                          xrange=xrange,yrange=yrange,xlabel=xlabel,
                          ylabel=ylabel,interpolation='nearest',
                          retCumImage=True,aspect=aspect,
                          overplot=onedhists)
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
    if not weights == None:
        w8= weights[cums > levels[-1]]
        for ii in range(len(plotx)):
            bovy_plot(plotx[ii],ploty[ii],overplot=True,
                      color='%.2f'%(1.-w8[ii]),*args,**kwargs)
    else:
        bovy_plot(plotx,ploty,overplot=True,*args,**kwargs)
    #Add onedhists
    if not onedhists:
        return
    axHistx.hist(x, bins=bins,normed=True,histtype=onedhisttype,range=xrange,
                 color=onedhistcolor,fc=onedhistfc,ec=onedhistec)
    axHisty.hist(y, bins=bins, orientation='horizontal',normed=True,
                 histtype=onedhisttype,range=yrange,
                 color=onedhistcolor,fc=onedhistfc,ec=onedhistec)
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

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
        
def _add_ticks():
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
    xstep= ax.xaxis.get_majorticklocs()
    xstep= xstep[1]-xstep[0]
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(xstep/5.))
    ystep= ax.yaxis.get_majorticklocs()
    ystep= ystep[1]-ystep[0]
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(ystep/5.))


    
