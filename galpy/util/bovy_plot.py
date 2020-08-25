# renamed to plot.py
from .plot import *
from .plot import _MPL_VERSION, _DEFAULTNCNTR, _add_axislabels, _add_ticks
import warnings
warnings.warn('galpy.util.bovy_plot is being deprecated in favor of galpy.util.plot; functions inside of this module have also changed name, but all functions still exist; please switch to the new import and new function names, because the old import and function names will be removed in v1.9',FutureWarning)
# Old names
bovy_end_print= end_print
bovy_hist= hist
bovy_plot= plot
bovy_plot3d= plot3d
bovy_dens2d= dens2d
bovy_print= start_print
bovy_text= text
