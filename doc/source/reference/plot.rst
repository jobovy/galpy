galpy.util.plot
====================

.. WARNING::
   Importing ``galpy.util.plot`` (or having it be imported by other ``galpy`` routines) with ``seaborn`` installed may change the ``seaborn`` plot style. If you don't like this, set the configuration parameter ``seaborn-plotting-defaults`` to False in the :ref:`configuration file <configfile>`

Various plotting routines:

.. toctree::
   :maxdepth: 1

   dens2d <plotdens2d.rst>
   end_print <plotendprint.rst>
   hist <plothist.rst>
   plot <plotplot.rst>
   print <plotstartprint.rst>
   text <plottext.rst>
   scatterplot <scatterplot.rst>

``galpy`` also contains a new matplotlib projection ``'galpolar'``
that can be used when working with older versions of matplotlib like
``'polar'`` to create a polar plot in which the azimuth increases
clockwise (like when looking at the Milky Way from the north Galactic
pole). In newer versions of matplotlib, this does not work, but the
``'polar'`` projection now supports clockwise azimuths by doing, e.g.,

>>> ax= pyplot.subplot(111,projection='polar')
>>> ax.set_theta_direction(-1)
