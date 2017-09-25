**NEW in v1.2**: What's new?
=============================

This page gives some of the key improvements in each galpy
version. See the ``HISTORY.txt`` file in the galpy source for full
details on what is new and different in each version.

v1.2
+++++

* Full support for providing inputs to all initializations, methods,
  and functions as `astropy Quantity
  <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__
  with `units <http://docs.astropy.org/en/stable/units/>`__ and for
  providing outputs as astropy Quantities.

* ``galpy.potential.TwoPowerTriaxialPotential``, a set of triaxial
  potentials with iso-density contours that are arbitrary, similar,
  coaxial ellipsoids whose 'radial' density is a (different) power-law
  at small and large radii: 1/m^alpha/(1+m)^beta-alpha (the triaxial
  generalization of TwoPowerSphericalPotential, with flattening in the
  density rather than in the potential; includes triaxial Hernquist
  and NFW potentials.

* ``galpy.potential.SCFPotential``, a class that implements general
  density/potential pairs through the basis expansion approach to
  solving the Poisson equation of Hernquist & Ostriker (1992).  Also
  implemented functions to compute the coefficients for a given
  density function. See more explanation :ref:`here
  <scf_potential_docs>`.

* ``galpy.actionAngle.actionAngleTorus``: an experimental interface to
  Binney & McMillan's TorusMapper code for computing positions and
  velocities for given actions and angles. See the installation
  instructions for how to properly install this. See :ref:`this
  section <aatorus>` and the ``galpy.actionAngle`` API page for
  documentation.

* ``galpy.actionAngle.actionAngleIsochroneApprox`` (Bovy 2014) now
  implemented for the general case of a time-independent potential.

* ``galpy.df.streamgapdf``, a module for modeling the effect of a
  dark-matter subhalo on a tidal stream. See `Sanders et al. (2016)
  <http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S>`__. Also
  includes the fast methods for computing the density along the stream
  and the stream track for a perturbed stream from `Bovy et al. (2016)
  <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1606.03470>`__.

* ``Orbit.flip`` can now flip the velocities of an orbit in-place by
  specifying ``inplace=True``. This allows correct velocities to be
  easily obtained for backwards-integrated orbits.

* ``galpy.potential.PseudoIsothermalPotential``, a standard
  pseudo-isothermal-sphere
  potential. ``galpy.potential.KuzminDiskPotential``, a razor-thin
  disk potential.

* Internal transformations between equatorial and Galactic coordinates
  are now performed by default using astropy's `coordinates
  <http://docs.astropy.org/en/stable/coordinates/index.html>`__
  module. Transformation of (ra,dec) to Galactic coordinates for
  general epochs.

v1.1
+++++

* Full support for Python 3.

* ``galpy.potential.SnapshotRZPotential``, a potential class that can
  be used to get a frozen snapshot of the potential of an N-body
  simulation.

* Various other potentials: ``PlummerPotential``, a standard Plummer
  potential; ``MN3ExponentialDiskPotential``, an approximation to an
  exponential disk using three Miyamoto-Nagai potentials (`Smith et
  al. 2015 <http://adsabs.harvard.edu/abs/2015MNRAS.448.2934S>`__);
  ``KuzminKutuzovStaeckelPotential``, a Staeckel potential that can be
  used to approximate the potential of a disk galaxy (`Batsleer &
  Dejonghe 1994
  <http://adsabs.harvard.edu/abs/1994A%26A...287...43B>`__).

* Support for converting potential parameters to `NEMO
  <http://bima.astro.umd.edu/nemo/>`__ format and units.

* Orbit fitting in custom sky coordinates.