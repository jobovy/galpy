galpy.util.bovy_plot
====================

.. WARNING::
   Importing ``galpy.util.bovy_plot`` (or having it be imported by other ``galpy`` routines) with ``seaborn`` installed will change the ``seaborn`` plot style. If you don't like this, call ``import seaborn; seaborn.set_style('darkgrid')`` (or similar for other styles) after importing ``galpy.util.bovy_plot``.

Various plotting routines:

.. toctree::
   :maxdepth: 2

   bovy_dens2d <bovydens2d.rst>
   bovy_end_print <bovyendprint.rst>
   bovy_hist <bovyhist.rst>
   bovy_plot <bovyplotplot.rst>
   bovy_print <bovyprint.rst>
   bovy_text <bovytext.rst>
   scatterplot <scatterplot.rst>

``galpy`` also contains a new matplotlib projection ``'galpolar'``
that can be used when working with older versions of matplotlib like
``'polar'`` to create a polar plot in which the azimuth increases
clockwise (like when looking at the Milky Way from the north Galactic
pole). In newer versions of matplotlib, this does not work, but the
``'polar'`` projection now supports clockwise azimuths by doing, e.g., 

>>> ax= pyplot.subplot(111,projection='polar')
>>> ax.set_theta_direction(-1)

