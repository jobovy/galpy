.. galpy documentation master file, created by
   sphinx-quickstart on Sun Jul 11 15:58:27 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to galpy's documentation
=================================

galpy is a python package for galactic dynamics. It supports orbit
integration in a variety of potentials, evaluating and sampling
various distribution functions, and the calculation of action-angle
coordinates for some potentials.

Quick-start guide
-----------------

.. toctree::
   :maxdepth: 2

   installation.rst

   getting_started.rst

   basic_df.rst

   orbit.rst

   diskdf.rst

   actionAngle.rst

Library reference
-----------------

.. toctree::
   :maxdepth: 2

   reference/orbit.rst

   reference/potential.rst

   reference/df.rst

   reference/util.rst

Papers using galpy
--------------------

Please let me (bovy -at- ias.edu) know if you make use of ``galpy`` in a publication.

* *Tracing the Hercules stream around the Galaxy*, Jo Bovy (2010), *Astrophys. J.* **725**, 1676 (`2010ApJ...725.1676B <http://adsabs.harvard.edu/abs/2010ApJ...725.1676B>`_): 
  	   Uses what later became the orbit integration routines and Dehnen and Shu disk distribution functions.
* *The spatial structure of mono-abundance sub-populations of the Milky Way disk*, Jo Bovy, Hans-Walter Rix, Chao Liu, et al. (2012), *Astrophys. J.* **753**, 148 (`2012ApJ...753..148B <http://adsabs.harvard.edu/abs/2012ApJ...753..148B>`_):
       Employs galpy orbit integration in ``galpy.potential.MWPotential`` to characterize the orbits in the SEGUE G dwarf sample.
* *On the local dark matter density*, Jo Bovy & Scott Tremaine (2012), *Astrophys. J.* **756**, 89 (`2012ApJ...756...89B <http://adsabs.harvard.edu/abs/2012ApJ...756...89B>`_):
      Uses ``galpy.potential`` force and density routines to characterize the difference between the vertical force and the surface density at large heights above the MW midplane.
* *The Milky Way's circular velocity curve between 4 and 14 kpc from APOGEE data*, Jo Bovy, Carlos Allende Prieto, Timothy C. Beers (2012), *Astrophys. J.* **759**, 131 (`2012ApJ...759..131B <http://adsabs.harvard.edu/abs/2012ApJ...759..131B>`_):
       Utilizes the Dehnen distribution function to inform a simple model of the velocity distribution of APOGEE stars in the Milky Way disk and to create mock data.
* *A direct dynamical measurement of the Milky Way's disk surface density profile, disk scale length, and dark matter profile at 4 kpc < R < 9 kpc*, Jo Bovy & Hans-Walter Rix (2013), *Astrophys. J.* **779**, 115 (`arXiv/1309.0809 <http://arxiv.org/abs/1309.0809>`_):
     Makes use of potential models, the adiabatic and Staeckel actionAngle modules, and the quasiisothermal DF to model the dynamics of the SEGUE G dwarf sample in mono-abundance bins.
* *Chemodynamics of the Milky Way. I. The first year of APOGEE data*, Friedrich Anders, Christina Chiappini, Basilio X. Santiago, et al. (2013), *Astron. & Astrophys.* submitted (`arXiv/1311.4549 <http://arxiv.org/abs/1311.4549>`_):
  		 Employs galpy to perform orbit integrations in ``galpy.potential.MWPotential`` to characterize the orbits of stars in the APOGEE sample.

Acknowledging galpy
--------------------

Please link back to ``http://code.google.com/p/galpy/`` . When using the ``galpy.actionAngle`` modules, please cite `arXiv/1309.0809 <http://arxiv.org/abs/1309.0809>`_ in addition to the papers describing the algorithm used. When orbit ingrations are used, you could cite `2010ApJ...725.1676B <http://adsabs.harvard.edu/abs/2010ApJ...725.1676B>`_ (first galpy paper).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

