.. galpy documentation master file, created by
   sphinx-quickstart on Sun Jul 11 15:58:27 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. ifconfig:: not_on_rtd

   .. WARNING:: You are looking at the rarely updated, GitHub version of this documentation, **please go to** `galpy.readthedocs.io <http://galpy.readthedocs.io>`_ **for the latest documentation**.

Welcome to galpy's documentation
=================================

galpy is a Python 2 and 3 package for galactic dynamics. It supports
orbit integration in a variety of potentials, evaluating and sampling
various distribution functions, and the calculation of action-angle
coordinates for all static potentials. galpy is an `astropy
<http://www.astropy.org/>`_ `affiliated package
<http://www.astropy.org/affiliated/>`_ and provides full support for
astropy's `Quantity
<http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`_
framework for variables with units.

galpy is developed on GitHub. If you are looking to `report an issue
<https://github.com/jobovy/galpy/issues>`_ or for information on how
to `contribute to the code
<https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors>`_,
please head over to `galpy's GitHub page
<https://github.com/jobovy/galpy>`_ for more information.

As a preview of the kinds of things you can do with galpy, here's an
:ref:`animation <orbanim>` of the orbit of the Sun in galpy's
``MWPotential2014`` potential over 7 Gyr:

.. raw:: html
   :file: orbitsun.html

Quick-start guide
-----------------

.. toctree::
   :maxdepth: 2

   installation.rst

   whatsnew.rst

   getting_started.rst

   potential.rst

   orbit.rst

   basic_df.rst

   actionAngle.rst

   diskdf.rst

Tutorials
---------

.. toctree::
   :maxdepth: 2

   streamdf.rst

Library reference
-----------------

.. toctree::
   :maxdepth: 2

   reference/orbit.rst

   reference/potential.rst

   reference/aa.rst

   reference/df.rst

   reference/util.rst


Acknowledging galpy
--------------------

If you use galpy in a publication, please cite the following paper

* *galpy: A Python Library for Galactic Dynamics*, Jo Bovy (2015), *Astrophys. J. Supp.*, **216**, 29 (`arXiv/1412.3451 <http://arxiv.org/abs/1412.3451>`_).

and link to ``http://github.com/jobovy/galpy``. Some of the code's
functionality is introduced in separate papers:

* ``galpy.actionAngle.EccZmaxRperiRap`` and ``galpy.orbit.Orbit`` methods with ``analytic=True``: Fast method for computing orbital parameters from :ref:`this section <fastchar>`: please cite `Mackereth & Bovy (2018) <https://arxiv.org/abs/1802.02592>`__.
* ``galpy.actionAngle.actionAngleAdiabatic``: please cite `Binney (2010) <http://adsabs.harvard.edu/abs/2010MNRAS.401.2318B>`__.
* ``galpy.actionAngle.actionAngleStaeckel``:  please cite `Bovy & Rix (2013) <http://adsabs.harvard.edu/abs/2013ApJ...779..115B>`__ and `Binney (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.426.1324B>`__.
* ``galpy.actionAngle.actionAngleIsochroneApprox``: please cite `Bovy (2014) <http://adsabs.harvard.edu/abs/2014ApJ...795...95B>`__.
* ``galpy.df.streamdf``: please cite `Bovy (2014) <http://adsabs.harvard.edu/abs/2014ApJ...795...95B>`__.
* ``galpy.df.streamgapdf``: please cite `Sanders, Bovy, & Erkal (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S>`__.
* ``galpy.potential.ttensor`` and ``galpy.potential.rtide``: please cite `Webb et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5748W/abstract>`__.

* ``galpy.potential.to_amuse``: please cite Webb et al. (2019; in prep).

Please also send me a reference to the paper or send a pull request
including your paper in the list of galpy papers on this page (this
page is at doc/source/index.rst). Thanks!

Papers using galpy
--------------------

``galpy`` is described in detail in this publication:

* *galpy: A Python Library for Galactic Dynamics*, Jo Bovy (2015), *Astrophys. J. Supp.*, **216**, 29 (`2015ApJS..216...29B <http://adsabs.harvard.edu/abs/2015ApJS..216...29B>`_).

The following is a list of publications using ``galpy``; please let me (bovy at astro dot utoronto dot ca) know if you make use of ``galpy`` in a publication.

#. *Tracing the Hercules stream around the Galaxy*, Jo Bovy (2010), *Astrophys. J.* **725**, 1676 (`2010ApJ...725.1676B <http://adsabs.harvard.edu/abs/2010ApJ...725.1676B>`_): 
  	   Uses what later became the orbit integration routines and Dehnen and Shu disk distribution functions.
#. *The spatial structure of mono-abundance sub-populations of the Milky Way disk*, Jo Bovy, Hans-Walter Rix, Chao Liu, et al. (2012), *Astrophys. J.* **753**, 148 (`2012ApJ...753..148B <http://adsabs.harvard.edu/abs/2012ApJ...753..148B>`_):
       Employs galpy orbit integration in ``galpy.potential.MWPotential`` to characterize the orbits in the SEGUE G dwarf sample.
#. *On the local dark matter density*, Jo Bovy & Scott Tremaine (2012), *Astrophys. J.* **756**, 89 (`2012ApJ...756...89B <http://adsabs.harvard.edu/abs/2012ApJ...756...89B>`_):
      Uses ``galpy.potential`` force and density routines to characterize the difference between the vertical force and the surface density at large heights above the MW midplane.
#. *The Milky Way's circular velocity curve between 4 and 14 kpc from APOGEE data*, Jo Bovy, Carlos Allende Prieto, Timothy C. Beers, et al. (2012), *Astrophys. J.* **759**, 131 (`2012ApJ...759..131B <http://adsabs.harvard.edu/abs/2012ApJ...759..131B>`_):
       Utilizes the Dehnen distribution function to inform a simple model of the velocity distribution of APOGEE stars in the Milky Way disk and to create mock data.
#. *A direct dynamical measurement of the Milky Way's disk surface density profile, disk scale length, and dark matter profile at 4 kpc < R < 9 kpc*, Jo Bovy & Hans-Walter Rix (2013), *Astrophys. J.* **779**, 115 (`2013ApJ...779..115B <http://adsabs.harvard.edu/abs/2013ApJ...779..115B>`_):
     Makes use of potential models, the adiabatic and Staeckel actionAngle modules, and the quasiisothermal DF to model the dynamics of the SEGUE G dwarf sample in mono-abundance bins.
#. *The peculiar pulsar population of the central parsec*, Jason Dexter & Ryan M. O'Leary (2013), *Astrophys. J. Lett.*, **783**, L7 (`2014ApJ...783L...7D <http://adsabs.harvard.edu/abs/2014ApJ...783L...7D>`_):
     Uses galpy for orbit integration of pulsars kicked out of the Galactic center.
#. *Chemodynamics of the Milky Way. I. The first year of APOGEE data*, Friedrich Anders, Christina Chiappini, Basilio X. Santiago, et al. (2013), *Astron. & Astrophys.*, **564**, A115 (`2014A&A...564A.115A <http://adsabs.harvard.edu/abs/2014A%26A...564A.115A>`_):
  		 Employs galpy to perform orbit integrations in ``galpy.potential.MWPotential`` to characterize the orbits of stars in the APOGEE sample.

#. *Dynamical modeling of tidal streams*, Jo Bovy (2014), *Astrophys. J.*, **795**, 95 (`2014ApJ...795...95B <http://adsabs.harvard.edu/abs/2014ApJ...795...95B>`_):
    Introduces ``galpy.df.streamdf`` and ``galpy.actionAngle.actionAngleIsochroneApprox`` for modeling tidal streams using simple models formulated in action-angle space (see the tutorial above).
#. *The Milky Way Tomography with SDSS. V. Mapping the Dark Matter Halo*, Sarah R. Loebman, Zeljko Ivezic Thomas R. Quinn, Jo Bovy, Charlotte R. Christensen, Mario Juric, Rok Roskar, Alyson M. Brooks, & Fabio Governato (2014), *Astrophys. J.*, **794**, 151 (`2014ApJ...794..151L <http://adsabs.harvard.edu/abs/2014ApJ...794..151L>`_):
    Uses ``galpy.potential`` functions to calculate the acceleration field of the best-fit potential in Bovy & Rix (2013) above.
#. *The Proper Motion of the Galactic Center Pulsar Relative to Sagittarius A**, Geoffrey C. Bower, Adam Deller, Paul Demorest, et al. (2015), *Astrophys. J.*, **798**, 120 (`2015ApJ...798..120B <http://adsabs.harvard.edu/abs/2015ApJ...798..120B>`_):
    Utilizes ``galpy.orbit`` integration in Monte Carlo simulations of the possible origin of the pulsar PSR J1745-2900 near the black hole at the center of the Milky Way.
#. *The power spectrum of the Milky Way: Velocity fluctuations in the Galactic disk*, Jo Bovy, Jonathan C. Bird, Ana E. Garcia Perez, Steven M. Majewski, David L. Nidever, & Gail Zasowski (2015), *Astrophys. J.*, **800**, 83 (`2015ApJ...800...83B <http://adsabs.harvard.edu/abs/2015ApJ...800...83B>`_):
    Uses ``galpy.df.evolveddiskdf`` to calculate the mean non-axisymmetric velocity field due to different non-axisymmetric perturbations and compares it to APOGEE data.
#. *The LMC geometry and outer stellar populations from early DES data*, Eduardo Balbinot, B. X. Santiago, L. Girardi, et al. (2015), *Mon. Not. Roy. Astron. Soc.*, **449**, 1129 (`2015MNRAS.449.1129B <http://adsabs.harvard.edu/abs/2015MNRAS.449.1129B>`_):
    Employs ``galpy.potential.MWPotential`` as a mass model for the Milky Way to constrain the mass of the LMC.
#. *Generation of mock tidal streams*, Mark A. Fardal, Shuiyao Huang, & Martin D. Weinberg (2015), *Mon. Not. Roy. Astron. Soc.*, **452**, 301 (`2015MNRAS.452..301F <http://adsabs.harvard.edu/abs/2015MNRAS.452..301F>`_):
    Uses ``galpy.potential`` and ``galpy.orbit`` for orbit integration in creating a *particle-spray* model for tidal streams.
#. *The nature and orbit of the Ophiuchus stream*, Branimir Sesar, Jo Bovy, Edouard J. Bernard, et al. (2015), *Astrophys. J.*, **809**, 59 (`2015ApJ...809...59S <http://adsabs.harvard.edu/abs/2015ApJ...809...59S>`_):
    Uses the ``Orbit.fit`` routine in ``galpy.orbit`` to fit the orbit of the Ophiuchus stream to newly obtained observational data and the routines in ``galpy.df.streamdf`` to model the creation of the stream.
#. *Young Pulsars and the Galactic Center GeV Gamma-ray Excess*, Ryan M. O'Leary, Matthew D. Kistler, Matthew Kerr, & Jason Dexter (2015), *Phys. Rev. Lett.*, submitted (`arXiv/1504.02477 <http://arxiv.org/abs/1504.02477>`_):
     Uses galpy orbit integration  and ``galpy.potential.MWPotential2014`` as part of a Monte Carlo simulation of the Galactic young-pulsar population.
#. *Phase Wrapping of Epicyclic Perturbations in the Wobbly Galaxy*, Alexander de la Vega, Alice C. Quillen, Jeffrey L. Carlin, Sukanya Chakrabarti, & Elena D'Onghia (2015), *Mon. Not. Roy. Astron. Soc.*, **454**, 933 (`2015MNRAS.454..933D <http://adsabs.harvard.edu/abs/2015MNRAS.454..933D>`_):
     Employs galpy orbit integration, ``galpy.potential`` functions, and ``galpy.potential.MWPotential2014`` to investigate epicyclic motions induced by the pericentric passage of a large dwarf galaxy and how these motions give rise to streaming motions in the vertical velocities of Milky Way disk stars.
#. *Chemistry of the Most Metal-poor Stars in the Bulge and the z ≳ 10 Universe*, Andrew R. Casey & Kevin C. Schlaufman (2015), *Astrophys. J.*, **809**, 110 (`2015ApJ...809..110C <http://adsabs.harvard.edu/abs/2015ApJ...809..110C>`_):
     This paper employs galpy orbit integration in ``MWPotential`` to characterize the orbits of three very metal-poor stars in the Galactic bulge.
#. *The Phoenix stream: a cold stream in the Southern hemisphere*, E. Balbinot, B. Yanny, T. S. Li, et al. (2015), *Astrophys. J.*, **820**, 58 (`2016ApJ...820...58B <http://adsabs.harvard.edu/abs/2016ApJ...820...58B>`_).
#. *Discovery of a Stellar Overdensity in Eridanus-Phoenix in the Dark Energy Survey*, T. S. Li, E. Balbinot, N. Mondrik, et al. (2015), *Astrophys. J.*, **817**, 135 (`2016ApJ...817..135L <http://adsabs.harvard.edu/abs/2016ApJ...817..135L>`_):
     Both of these papers use galpy orbit integration to integrate the orbit of NGC 1261 to investigate a possible association of this cluster with the newly discovered Phoenix stream and Eridanus-Phoenix overdensity.
#. *The Proper Motion of Palomar 5*, T. K. Fritz & N. Kallivayalil (2015), *Astrophys. J.*, **811**, 123 (`2015ApJ...811..123F <http://adsabs.harvard.edu/abs/2015ApJ...811..123F>`_):
     This paper makes use of the ``galpy.df.streamdf`` model for tidal streams to constrain the Milky Way's gravitational potential using the kinematics of the Palomar 5 cluster and stream.
#. *Spiral- and bar-driven peculiar velocities in Milky Way-sized galaxy simulations*, Robert J. J. Grand, Jo Bovy, Daisuke Kawata, Jason A. S. Hunt, Benoit Famaey, Arnaud Siebert, Giacomo Monari, & Mark Cropper (2015), *Mon. Not. Roy. Astron. Soc.*, **453**, 1867 (`2015MNRAS.453.1867G <http://adsabs.harvard.edu/abs/2015MNRAS.453.1867G>`_):
     Uses ``galpy.df.evolveddiskdf`` to calculate the mean non-axisymmetric velo\city field due to the bar in different parts of the Milky Way.
#. *Vertical kinematics of the thick disc at 4.5 ≲ R ≲ 9.5 kpc*, Kohei Hattori & Gerard Gilmore (2015), *Mon. Not. Roy. Astron. Soc.*, **454**, 649 (`2015MNRAS.454..649H <http://adsabs.harvard.edu/abs/2015MNRAS.454..649H>`_):
     This paper uses ``galpy.potential`` functions to set up a realistic Milky-Way potential for investigating the kinematics of stars in the thick disk.
#. *Local Stellar Kinematics from RAVE data - VI. Metallicity Gradients Based on the F-G Main-sequence Stars*, O. Plevne, T. Ak, S. Karaali, S. Bilir, S. Ak, Z. F. Bostanci (2015), *Pub. Astron. Soc. Aus.*, **32**, 43 (`2015PASA...32...43P <http://adsabs.harvard.edu/abs/2015PASA...32...43P>`_):
     This paper employs galpy orbit integration in ``MWPotential2014`` to calculate orbital parameters for a sample of RAVE F and G dwarfs to investigate the metallicity gradient in the Milky Way.
#. *Dynamics of stream-subhalo interactions*, Jason L. Sanders, Jo Bovy, & Denis Erkal (2015), *Mon. Not. Roy. Astron. Soc.*, **457**, 3817 (`2016MNRAS.457.3817S <http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S>`_):
     Uses and extends ``galpy.df.streamdf`` to build a generative model of the dynamical effect of sub-halo impacts on tidal streams. This new functionality is contained in ``galpy.df.streamgapdf``, a subclass of ``galpy.df.streamdf``, and can be used to efficiently model the effect of impacts on the present-day structure of streams in position and velocity space.
#. *Extremely metal-poor stars from the cosmic dawn in the bulge of the Milky Way*, L. M. Howes, A. R. Casey, M. Asplund, et al. (2015), *Nature*, **527**, 484 (`2015Natur.527..484H <http://adsabs.harvard.edu/abs/2015Natur.527..484H>`_):
     Employs galpy orbit integration in ``MWPotential2014`` to characterize the orbits of a sample of extremely metal-poor stars found in the bulge of the Milky Way. This analysis demonstrates that the orbits of these metal-poor stars are always close to the center of the Milky Way and that these stars are therefore true bulge stars rather than halo stars passing through the bulge.
#. *Detecting the disruption of dark-matter halos with stellar streams*, Jo Bovy (2016), *Phys. Rev. Lett.*, **116**, 121301 (`2016PhRvL.116l1301B <http://adsabs.harvard.edu/abs/2016PhRvL.116l1301B>`_):
     Uses galpy functions in ``galpy.df`` to estimate the velocity kick imparted by a disrupting dark-matter halo on a stellar stream. Also employs ``galpy.orbit`` integration and ``galpy.actionAngle`` functions to analyze *N*-body simulations of such an interaction.
#. *Identification of Globular Cluster Stars in RAVE data II: Extended tidal debris around NGC 3201*, B. Anguiano, G. M. De Silva, K. Freeman, et al. (2016), *Mon. Not. Roy. Astron. Soc.*, **457**, 2078 (`2016MNRAS.457.2078A <http://adsabs.harvard.edu/abs/2016MNRAS.457.2078A>`_):
     Employs ``galpy.orbit`` integration to study the orbits of potential tidal-debris members of NGC 3201.
#. *Young and Millisecond Pulsar GeV Gamma-ray Fluxes from the Galactic Center and Beyond*, Ryan M. O'Leary, Matthew D. Kistler, Matthew Kerr, & Jason Dexter (2016), *Phys. Rev. D*, submitted (`arXiv/1601.05797 <http://arxiv.org/abs/1601.05797>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` for orbit integration of pulsars kicked out of the central region of the Milky Way.
#. *Abundances and kinematics for ten anticentre open clusters*, T. Cantat-Gaudin, P. Donati, A. Vallenari, R. Sordo, A. Bragaglia, L. Magrini (2016), *Astron. & Astrophys.*, **588**, A120 (`2016A&A...588A.120C <http://adsabs.harvard.edu/abs/2016A%26A...588A.120C>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to characterize the orbits of 10 open clusters located toward the Galactic anti-center, finding that the most distant clusters have high-eccentricity orbits.
#. *A Magellanic Origin of the DES Dwarfs*, P. Jethwa, D. Erkal, & V. Belokurov (2016), *Mon. Not. Roy. Astron. Soc.*, **461**, 2212 (`arXiv/1603.04420 <http://arxiv.org/abs/1603.04420>`_):
     Employs the C implementations of ``galpy.potential``\s to compute forces in orbit integrations of the LMC's satellite-galaxy population.
#. *PSR J1024-0719: A Millisecond Pulsar in an Unusual Long-Period Orbit*, D. L. Kaplan, T. Kupfer, D. J. Nice, et al. (2016), *Astrophys. J.*, **826**, 86 (`arXiv/1604.00131 <http://arxiv.org/abs/1604.00131>`_):
#. *A millisecond pulsar in an extremely wide binary system*, C. G. Bassa, G. H. Janssen, B. W. Stappers, et al. (2016), *Mon. Not. Roy. Astron. Soc.*, **460**, 2207 (`arXiv/1604.00129 <http://arxiv.org/abs/1604.00129>`_):
     Both of these papers use ``galpy.orbit`` integration in ``MWPotential2014`` to determine the orbit of the milli-second pulsar PSR J1024−0719, a pulsar in an unusual binary system.
#. *The first low-mass black hole X-ray binary identified in quiescence outside of a globular cluster*, B. E. Tetarenko, A. Bahramian, R. M. Arnason, et al. (2016), *Astrophys. J.*, **825**, 10 (`arXiv/1605.00270 <http://arxiv.org/abs/1605.00270>`_):
     This paper employs ``galpy.orbit`` integration of orbits within the position-velocity uncertainty ellipse of the radio source VLA J213002.08+120904 to help characterize its nature (specifically, to rule out that it is a magnetar based on its birth location).
#. *Action-based Dynamical Modelling for the Milky Way Disk*, Wilma H. Trick, Jo Bovy, & Hans-Walter Rix (2016), *Astrophys. J.*, **830**, 97 (`arXiv/1605.08601 <http://arxiv.org/abs/1605.08601>`_):
     Makes use of potential models, the Staeckel actionAngle modules, and the quasiisothermal DF to develop a robust dynamical modeling approach for recovering the Milky Way's gravitational potential from kinematics of disk stars.
#. *A Dipole on the Sky: Predictions for Hypervelocity Stars from the Large Magellanic Cloud*, Douglas Boubert & N. W. Evans (2016), *Astrophys. J. Lett.*, **825**, L6 (`arXiv/1606.02548 <http://arxiv.org/abs/1606.02548>`_):
     Uses ``galpy.orbit`` integration to investigate the orbits of hyper-velocity stars that could be ejected from the Large Magellanic Cloud and their distribution on the sky.
#. *Linear perturbation theory for tidal streams and the small-scale CDM power spectrum*, Jo Bovy, Denis Erkal, & Jason L. Sanders (2016), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1606.03470 <http://arxiv.org/abs/1606.03470>`_):
     Uses and extends ``galpy.df.streamdf`` and ``galpy.df.streamgapdf`` to quickly compute the effect of impacts from dark-matter subhalos on stellar streams and investigates the structure of perturbed streams and how this structure relates to the CDM subhalo mass spectrum.
#. *Local Stellar Kinematics from RAVE data - VII. Metallicity Gradients from Red Clump Stars*, O. Onal Tas, S. Bilir, G. M. Seabroke, S. Karaali, S. Ak, T. Ak, & Z. F. Bostanci (2016), *Pub. Astron. Soc. Aus.*, **33**, e044 (`arXiv/1607.07049 <http://arxiv.org/abs/1607.07049>`_):
     Employs galpy orbit integration in ``MWPotential2014`` to calculate orbital parameters for a sample of red clump stars in RAVE to investigate the metallicity gradient in the Milky Way.
#. *Study of Eclipsing Binary and Multiple Systems in OB Associations IV: Cas OB6 Member DN Cas*, V. Bakis, H. Bakis, S. Bilir, Z. Eker (2016), *Pub. Astron. Soc. Aus.*, **33**, e046 (`arXiv/1608.00456 <http://arxiv.org/abs/1608.00456>`_):
     Uses galpy orbit integration in ``MWPotential2014`` to calculate the orbit and orbital parameters of the spectroscopic binary DN Cas in the Milky Way.
#. *The shape of the inner Milky Way halo from observations of the Pal 5 and GD-1 stellar streams*, Jo Bovy, Anita Bahmanyar, Tobias K. Fritz, & Nitya Kallivayalil (2016), *Astrophys. J.*, in press (`arXiv/1609.01298 <http://arxiv.org/abs/1609.01298>`_):
     Makes use of the ``galpy.df.streamdf`` model for a tidal stream to constrain the shape and mass of the Milky Way's dark-matter halo. Introduced ``galpy.potential.TriaxialNFWPotential``.
#. *The Rotation-Metallicity Relation for the Galactic Disk as Measured in the Gaia DR1 TGAS and APOGEE Data*, Carlos Allende Prieto, Daisuke Kawata, & Mark Cropper (2016), *Astron. & Astrophys.*, in press (`arXiv/1609.07821 <http://arxiv.org/abs/1609.07821>`_):
     Employs orbit integration in ``MWPotential2014`` to calculate the orbits of a sample of stars in common between Gaia DR1's TGAS and APOGEE to study the rotation-metallicity relation for the Galactic disk.
#. *Detection of a dearth of stars with zero angular momentum in the solar neighbourhood*, Jason A. S. Hunt, Jo Bovy, & Raymond Carlberg (2016), *Astrophys. J. Lett.*, **832**, L25 (`arXiv/1610.02030 <http://arxiv.org/abs/1610.02030>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` plus a hard Galactic core to calculate the orbits of stars in the solar neighborhood and predict how many of them should be lost to chaos.
#. *Differences in the rotational properties of multiple stellar populations in M 13: a faster rotation for the "extreme" chemical subpopulation*, M. J. Cordero, V. Hénault-Brunet, C. A. Pilachowski, E. Balbinot, C. I. Johnson, & A. L. Varri (2016), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1610.09374 <http://arxiv.org/abs/1610.09374>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit of the globular cluster M13 and in particular whether escaping stars from the cluster could contaminate the measurement of the rotation of different populations in the cluster.
#. *Using the Multi-Object Adaptive Optics demonstrator RAVEN to observe metal-poor stars in and towards the Galactic Centre*, Masen Lamb, Kim Venn, David Andersen, et al. (2016), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1611.02712 <http://arxiv.org/abs/1611.02712>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to characterize the orbits of three very metal-poor stars observed toward the Galactic center, to determine whether they are likely bulge members.
#. *The Radial Velocity Experiment (RAVE): Fifth Data Release*, Andrea Kunder, Georges Kordopatis, Matthias Steinmetz, et al. (2016), *Astron. J.*, in press (`arXiv/1609.03210 <http://arxiv.org/abs/1609.03210>`_):
     Employs ``galpy.orbit`` integration to characterize the orbits of stars in the RAVE survey.
#. *The Proper Motion of Pyxis: the first use of Adaptive Optics in tandem with HST on a faint halo object*, Tobias K. Fritz, Sean Linden, Paul Zivick, et al. (2016), *Astrophys. J.*, submitted (`arXiv/1611.08598 <http://arxiv.org/abs/1611.08598>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit of the globular cluster Pyxis using its newly measured proper motion and to search for potential streams associated with the cluster.
#. *The Galactic distribution of X-ray binaries and its implications for compact object formation and natal kicks*, Serena Repetto, Andrei P. Igoshev, & Gijs Nelemans (2017), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1701.01347 <http://arxiv.org/abs/1701.01347>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` and that of `Paczynski (1990) <http://adsabs.harvard.edu/abs/1990ApJ...348..485P>`__ to study the orbits of X-ray binaries under different assumptions about their formation mechanism and natal velocity kicks.
#. *Kinematics of Subluminous O and B Stars by Surface Helium Abundance*, P. Martin, C. S. Jeffery, Naslim N., & V. M. Woolf (2017), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1701.03026 <http://arxiv.org/abs/1701.03026>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbits of different types of low-mass core-helium-burning stars.
#. *Is there a disk of satellites around the Milky Way?*, Moupiya Maji, Qirong Zhu, Federico Marinacci, & Yuexing Li (2017), submitted (`arXiv/1702.00485 <http://arxiv.org/abs/1702.00485>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to predict the future paths of 11 classical Milky-Way satellites to investigate whether they remain in a disk configuration.
#. *The devil is in the tails: the role of globular cluster mass evolution on stream properties*, Eduardo Balbinot & Mark Gieles (2017), *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1702.02543 <http://arxiv.org/abs/1702.02543>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` of globular clusters in the Milky-Way halo. These integrations are used to investigate the clusters'  mass loss due to tidal stripping, taking the effects of collisional dynamics in the cluster into account, and to evaluate the visibility of their (potential) tidal tails.
#. *Absolute Ages and Distances of 22 GCs using Monte Carlo Main-Sequence Fitting*, Erin M. O'Malley, Christina Gilligan, & Brian Chaboyer (2017), *Astrophys. J.*, in press (`arXiv/1703.01915 <http://arxiv.org/abs/1703.01915>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` of globular clusters in the Milky Way, to study their orbits and classify them as disk or halo clusters.
#. Siriusly, *a newly identified intermediate-age Milky Way stellar cluster: A spectroscopic study of* Gaia *1*, J. D. Simpson, G. M. De Silva, S. L. Martell, D. B. Zucker, A. M. N. Ferguson, E. J. Bernard, M. Irwin, J. Penarrubia, & E. Tolstoy (2017), *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1703.03823 <http://arxiv.org/abs/1703.03823>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit in the Milky Way potential of a newly-confirmed stellar cluster found in the Gaia data.
#. *Action-based Dynamical Modeling for the Milky Way Disk: The Influence of Spiral Arms*, Wilma H. Trick, Jo Bovy, Elena D'Onghia, & Hans-Walter Rix (2017), *Astrophys. J., in press* (`arXiv/1703.05970 <http://arxiv.org/abs/1703.05970>`_):
     Uses various potential models, the Staeckel actionAngle modules, and the quasiisothermal DF to test a robust dynamical modeling approach for recovering the Milky Way's gravitational potential from kinematics of disk stars against numerical simulations with spiral arms.
#. *A spectroscopic study of the elusive globular cluster ESO452-SC11 and its surroundings*, Andreas Koch, Camilla Juul Hansen, & Andrea Kunder (2017), *Astron. & Astrophys.*, submitted (`arXiv/1703.06921 <http://arxiv.org/abs/1703.06921>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit in the Milky Way potential of two candidate cluster members of the bulge globular cluster ESO452-SC11.
#. *A Halo Substructure in Gaia Data Release 1*, G. C. Myeong, N. W. Evans, V. Belokurov, S. E. Koposov, & J. L. Sanders (2017), *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1704.01363 <http://arxiv.org/abs/1704.01363>`_):
     Uses ``galpy.actionAngle.actionAngleAdiabatic`` routines to compute the actions using the adiabatic approximation for 268,588 stars in *Gaia* DR1 *TGAS* with line-of-sight velocities from spectroscopic surveys. Detects a co-moving group of 14 stars on strongly radial orbits and computes their orbits using ``MWPotential2014``.
#. *An artificial neural network to discover Hypervelocity stars: Candidates in* Gaia *DR1/* TGAS, T. Marchetti, E. M. Rossi, G. Kordopatis, A. G. A. Brown, A. Rimoldi, E. Starkenburg, K. Youakim, & R. Ashley (2017), *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1704.07990 <http://arxiv.org/abs/1704.07990>`_):
     Uses ``galpy.orbit`` integration in a custom Milky-Way-like potential built from ``galpy.potential`` models to investigate the orbits of hypervelocity-star candidates in *Gaia* DR1.
#. *GalRotpy: an educational tool to understand and parametrize the rotation curve and gravitational potential of disk-like galaxies*, Andrés Granados, Lady-J. Henao-O., Santiago Vanegas, & Leonardo Castañeda (2017; `arXiv/1705.01665 <http://arxiv.org/abs/1705.01665>`_):
     These authors build an interactive tool to decompose observed rotation curves into bulge, disk (Miyamoto-Nagai or exponential), and NFW halo components on top of ``galpy.potential`` routines.
#. *The AMBRE Project: formation and evolution of the Milky Way disc*, V. Grisoni, E. Spitoni, F. Matteucci, A. Recio-Blanco, P. de Laverny, M. Hayden, S. Mikolaitis, & C. C. Worley (2017) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1706.02614 <http://arxiv.org/abs/1706.02614>`_):
     Uses ``galpy`` to compute orbital parameters for stars in the AMBRE sample of high-resolution spectra and uses these orbital parameters to aid in the comparison between the data and chemical-evolution models.
#. *Low-mass X-ray binaries from black-hole retaining globular clusters*, Matthew Giesler, Drew Clausen, & Christian D. Ott (2017) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1708.05915 <http://arxiv.org/abs/1708.05915>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to compute the orbits of MW globular clusters and simulated ejected BH low-mass X-ray binaries from these globular clusters to determine where they end up today.
#. *ESO452-SC11: The lowest mass globular cluster with a potential chemical inhomogeneity*, Jeffrey D. Simpson, Gayandhi De Silva, Sarah L. Martell, Colin A. Navin, & Daniel B. Zucker (2017) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1708.06875 <http://arxiv.org/abs/1708.06875>`_):
     Uses ``galpy.orbit`` in ``MWPotential2014`` to compute the orbit of the MW bulge globular cluster ESO452-SC11.
#. *The Hercules stream as seen by APOGEE-2 South*, Jason A. S. Hunt, Jo Bovy, Angeles Pérez-Villegas, et al. (2017) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1709.02807 <http://arxiv.org/abs/1709.02807>`_):
     Uses ``galpy.df.evolveddiskdf`` to compute the effect of a fast-rotating Milky-Way bar model on the velocity distribution in the disk of the Milky Way near the Sun to compare to the observed line-of-sight velocity distribution in APOGEE-2 South.
#. *Detailed chemical abundance analysis of the thick disk star cluster Gaia 1*, Andreas Koch, Terese T. Hansen, & Andrea Kunder (2017) *Astron. & Astrophys.*, in press (`arXiv/1709.04022 <http://arxiv.org/abs/1709.04022>`_):
     Employs ``galpy.orbit`` integration to compute the orbits of four red-giant members of the *Gaia 1* Milky Way star cluster, finding that the orbits of these stars are similar to those of the oldest stars in the Milky Way's disk.
#. *Proper motions in the VVV Survey: Results for more than 15 million stars across NGC 6544*, R. Contreras Ramos, M. Zoccali, F. Rojas, A. Rojas-Arriagada, M. Gárate, P. Huijse, F. Gran, M. Soto, A.A.R. Valcarce, P. A. Estévez, & D. Minniti (2017) *Astron. & Astrophys.*, in press (`arXiv/1709.07919 <http://arxiv.org/abs/1709.07919>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to calculate the orbit of NGC 6544, a Milky-Way globular cluster, using a newly determined proper motion, finding that it is likely a halo globular cluster based on its orbit.
#. *How to make a mature accreting magnetar*, A. P. Igoshev & S. B. Popov (2017) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1709.10385 <http://arxiv.org/abs/1709.10385>`_):
     Employs ``galpy.orbit`` integration of the magnetar candidate 4U 0114+65 in the potential model from `Irrgang et al. (2013) <http://adsabs.harvard.edu/abs/2013A%26A...549A.137I>`__ to aid in the determination of its likely age.
#. *iota Horologii is unlikely to be an evaporated Hyades star*, I. Ramirez, D. Yong, E. Gutierrez, M. Endl, D. L. Lambert, J.-D. Do Nascimento Jr (2017) *Astrophys. J.*, in press (`arXiv/1710.05930 <http://arxiv.org/abs/1710.05930>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to determine the approximate orbit of the star iota Horologii, a planet-hosting suspected former member of the Hyades cluster, to investigate whether it could have coincided with the Hyades cluster in the past.
#. *Confirming chemical clocks: asteroseismic age dissection of the Milky Way disk(s)*, V. Silva Aguirre, M. Bojsen-Hansen, D. Slumstrup, et al. (2017) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1710.09847 <http://arxiv.org/abs/1710.09847>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to compute the orbits of a sample of 1989 red giants with spectroscopic and asteroseismic data from the APOKASC catalog, to shed light on the properties of stellar populations defined by age and metallicity.
#. *The universality of the rapid neutron-capture process revealed by a possible disrupted dwarf galaxy star*, Andrew R. Casey & Kevin C. Schlaufman (2017) *Astrophys. J.*, in press (`arXiv/1711.04776 <http://arxiv.org/abs/1711.04776>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit and its uncertainty of 2MASS J151113.24–213003.0, an extremely metal-poor field star with measureable r-process abundances, and of other similar metal-poor stars. The authors find that all of these stars are on highly eccentric orbits, possibly indicating that they originated in dwarf galaxies.
#. *The Gaia-ESO Survey: Churning through the Milky Way*, M. R. Hayden, A. Recio-Blanco, P. de Laverny, et al. (2017) *Astron. & Astrophys.*, in press (`arXiv/1711.05751 <http://arxiv.org/abs/1711.05751>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to study the orbital characteristics (eccentricity, pericentric radius) of a sample of 2,364 stars observed in the Milky Way as part of the Gaia-ESO survey.
#. *The Evolution of the Galactic Thick Disk with the LAMOST Survey*, Chengdong Li & Gang Zhao (2017) *Astrophys. J.*, **850**, 25 (`2017ApJ...850...25L <http://adsabs.harvard.edu/abs/2017ApJ...850...25L>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbital characteristics (eccentricity, maximum height above the plane, angular momentum) of a sample of about 2,000 stars in the thicker-disk component of the Milky Way.
#. *The Orbit and Origin of the Ultra-faint Dwarf Galaxy Segue 1*, T. K. Fritz, M. Lokken, N. Kallivayalil, A. Wetzel, S. T. Linden, P. Zivick, & E. J. Tollerud (2017) *Astrophys. J.*, submitted (`arXiv/1711.09097 <http://arxiv.org/abs/1711.09097>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` and a version of this potential with a more massive dark-matter halo to investigate the orbit and origin of the dwarf-spheroidal galaxy Segue 1 using a newly measured proper motion with SDSS and LBC data.
#. *Prospects for detection of hypervelocity stars with Gaia*, T. Marchetti, O. Contigiani, E. M. Rossi, J. G. Albert, A. G. A. Brown, & A. Sesana (2017) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1711.11397 <http://arxiv.org/abs/1711.11397>`_):
     Uses ``galpy.orbit`` integration in a custom Milky-Way-like potential built from ``galpy.potential`` models to create mock catalogs of hypervelocity stars in the Milky Way for different ejection mechanisms and study the prospects of their detection with *Gaia*.
#. *The AMBRE project: The thick thin disk and thin thick disk of the Milky Way*, Hayden, M. R., Recio-Blanco, A., de Laverny, P., Mikolaitis, S., & Worley, C. C. (2017) *Astron. & Astrophys.*, **608**, L1 (`arXiv/1712.02358 <http://arxiv.org/abs/1712.02358>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to characterize the orbits of 494 nearby stars analyzed as part of the AMBRE project to learn about their distribution within the Milky Way.
#. *KELT-21b: A Hot Jupiter Transiting the Rapidly-Rotating Metal-Poor Late-A Primary of a Likely Hierarchical Triple System*, Marshall C. Johnson, Joseph E. Rodriguez, George Zhou, et al. (2017) *Astrophys. J.*, submitted (`arXiv/1712.03241 <http://arxiv.org/abs/1712.03241>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the Galactic orbit of KELT-21b, a hot jupiter around a low-metallicity A-type star.
#. *GalDynPsr: A package to estimate dynamical contributions in the rate of change of the period of radio pulsars*, Dhruv Pathak & Manjari Bagchi (2017) (`arXiv/1712.06590 <http://arxiv.org/abs/1712.06590>`_):
     Presents a python package to compute contributions to the GR spin-down of pulsars from the differential galactic acceleration between the Sun and the pulsar. The package uses ``MWPotential2014`` and ``galpy.potential`` functions to help compute this.
#. *Local Stellar Kinematics from RAVE data – VIII. Effects of the Galactic Disc Perturbations on Stellar Orbits of Red Clump Stars*, O. Onal Tas, S. Bilir, &  O. Plevne (2018) *Astrophys. Sp. Sc.*, in press (`arXiv/1801.02170 <http://arxiv.org/abs/1801.02170>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` and the non-axisymmetric ``DehnenBarPotential`` and ``SteadyLogSpiralPotential`` to study the orbits of Milky-Way red-clump stars.
#. *The VMC survey XXVIII. Improved measurements of the proper motion of the Galactic globular cluster 47 Tucanae*, F. Niederhofer, M.-R. L. Cioni, S. Rubele, et al. (2018) *Astron. & Astrophys.*, in press (`arXiv/1801.07738 <http://arxiv.org/abs/1801.07738>`_):
     Uses ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbit of the cluster 47 Tuc from a newly measured proper motion, finding that the orbit has an eccentricity of about 0.2 and reaches up to 3.6 kpc above the Galactic midplane.
#. *Characterising Open Clusters in the solar neighbourhood with the Tycho-Gaia Astrometric Solution*, T. Cantat-Gaudin, A. Vallenari, R. Sordo, F. Pensabene, A. Krone-Martins, A. Moitinho, C. Jordi, L. Casamiquela, L. Balaguer-Núnez, C. Soubiran, N. Brouillet (2018) *Astron. & Astrophys.*, submitted (`arXiv/1801.10042 <http://arxiv.org/abs/1801.10042>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to investigate the orbits of a sample of 128 open clusters with astrometry from the Tycho-*Gaia* Astrometric Solution.
#. *Fast estimation of orbital parameters in Milky-Way-like potentials*, J. Ted Mackereth & Jo Bovy (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1802.02592 <http://arxiv.org/abs/1802.02592>`_):
     Introduces the ``galpy.actionAngle.EccZmaxRperiRap`` and related ``galpy.orbit.Orbit`` methods for the fast estimation of the orbital parameters (eccentricity, maximum vertical excursion, and peri/apocenter) in Milky-Way potentials. See :ref:`this section <fastchar>` of the documentation for more info.
#. *HI Kinematics Along The Minor Axis of M82*, Paul Martini, Adam K. Leroy, Jeffrey G. Mangum, Alberto Bolatto, Katie M. Keating, Karin Sandstrom, & Fabian Walter (2018) *Astrophys. J.*, in press (`arXiv/1802.04359 <http://arxiv.org/abs/1802.04359>`_):
     Use ``galpy.potential`` components to create a mass model for M82 that consists of a ``HernquistPotential`` bulge, ``MN3ExponentialDiskPotential`` disk, and ``NFWPotential`` dark-matter halo by matching photometric and rotation-curve data.
#. *Radial velocities of RR Lyrae stars in and around NGC 6441*, Andrea Kunder, Arthur Mills, Joseph Edgecomb, et al. (2018) *Astron. J.*, in press (`arXiv/1802.09562 <http://arxiv.org/abs/1802.09562>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to investigate a possible tidal stream for the globular cluster NGC 6441.
#. *The 4:1 Outer Lindblad Resonance of a long slow bar as a potential explanation for the Hercules stream*, Jason A. S. Hunt & Jo Bovy (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1803.02358 <http://arxiv.org/abs/1803.02358>`_):
     Uses ``galpy.potential`` modeling tools (``CosmphiDiskPotential`` and wrappers to grow it [``DehnenSmoothWrapperPotential``] and make it rotate [``SolidBodyRotationWrapperPotential``]) and disk models in ``galpy.df`` to study the effect of the Milky Way bar's hexadecapole (m=4) moment on the velocity distribution of stars in the disk of the Milky Way, finding that it can have a big effect near the 4:1 outer Lindblad resonance, which may be close to the Sun.
#. *KELT-22Ab: A Massive Hot Jupiter Transiting a Near Solar Twin*, Jonathan Labadie-Bartz, Joseph E. Rodriguez, Keivan G. Stassun, et al. (2018) *Astrophys. J.*, submitted (`arXiv/1803.07559 <http://arxiv.org/abs/1803.07559>`_):
     Employs ``galpy.orbit`` integration in ``MWPotential2014`` to explore the orbit of KELT-22A, the G2V host star of a transiting hot-jupiter exoplanet, in the Milky Way.
#. *Dissecting stellar chemical abundance space with t-SNE*, Friedrich Anders, Cristina Chiappini, Basílio X. Santiago, Gal Matijevič, Anna B. Queiroz, Matthias Steinmetz (2018) *Astron. & Astrophys.*, submitted (`arXiv/1803.09341 <http://arxiv.org/abs/1803.09341>`_):
     Uses the ``galpy.actionAngle.EccZmaxRperiRap`` and related ``galpy.orbit.Orbit`` methods introduced in `Mackereth & Bovy (2018) <https://arxiv.org/abs/1802.02592>`__ to compute the eccentricity and maximum vertical excursion of a sample of solar-neighborhood stars with detailed chemical abundances, ages, and Galactic kinematics.
#. *Chemo-kinematics of the Milky Way from the SDSS-III MARVELS Survey*, Nolan Grieves, Jian Ge, Neil Thomas, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1803.11538 <http://arxiv.org/abs/1803.11538>`_):
     Employs ``galpy.orbit`` integration in a Milky-Way like potential (Miyamoto-Nagai disk, Hernquist bulge, and NFW dark-matter halo) to determine the eccentricity, maximum vertical excursion, and median Galactocentric radius of a sample of a few thousand solar neighborhood stars with spectra from the SDSS MARVELS survey.
#. *The demographics of neutron star - white dwarf mergers: rates, delay-time distributions and progenitors*, S. Toonen, H.B. Perets, A.P. Igoshev, E. Michaely, & Y. Zenati (2018) *Astron. & Astrophys.*, submitted (`arXiv/1804.01538 <http://arxiv.org/abs/1804.01538>`_):
     Uses ``galpy.potential`` methods to build mass models for elliptical, dwarf elliptical, plus normal and bulgeless disk galaxies and ``galpy.orbit`` integration to integrate the orbits of neutron-star--white-dwarf binaries kicked by the supernova explosion that created the neutron star, to investigate the spatial distribution of such binaries at the time of their mergers.
#. *Absolute HST Proper Motion (HSTPROMO) Catalog of Distant Milky Way Globular Clusters: Three-dimensional Systemic Velocities and the Milky Way Mass*, Sangmo Tony Sohn, Laura L. Watkins, Mark A. Fardal, Roeland P. van der Marel, Alis J. Deason, Gurtina Besla, & Andrea Bellini (2018) *Astrophys. J.*, submitted (`arXiv/1804.01994 <http://arxiv.org/abs/1804.01994>`_):
     Employs ``galpy.orbit`` integration in a custom Milky-Way-like potential (a scaled version of ``MWPotential2014``) to study the orbits of 20 globular clusters in the Milky Way halo for which the authors measured new proper motions.
#. *Probing the nature of dark matter particles with stellar streams*, Nilanjan Banik, Gianfranco Bertone, Jo Bovy, & Nassim Bozorgnia (2018) *J. Cosmol. Astropart. Phys.*, submitted (`arXiv/1804.04384 <http://arxiv.org/abs/1804.04384>`_):
     Uses ``galpy.df.streamdf`` and the `galpy extension <https://gist.github.com/jobovy/1be0be25b525e5f50ea3>`_ ``galpy.df.streampepperdf`` to compute the effect of impacts from dark-matter subhalos on stellar streams when dark matter is warm, when there are many fewer such subhalos, and compares this to the standard CDM case.
#. *Connecting the Milky Way potential profile to the orbital timescales and spatial structure of the Sagittarius Stream*, Mark A. Fardal, Roeland P. van der Marel, David R. Law, Sangmo Tony Sohn, Branimir Sesar, Nina Hernitschek, & Hans-Walter Rix (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1804.04995 <http://arxiv.org/abs/1804.04995>`_):
     Employs the ``galpy.potential`` module to build models for the Milky Way's gravitational potential and uses ``galpy.orbit`` to integrate the orbits of members of the Sgr stream in a particle-spray model of the Sgr stream, to investigate how the Sgr stream is sensitive to the Milky Way's potential.

     At this point, a flood of papers using ``galpy`` started to appear, which we will not attempt to summarize:

#. *Evidence for accreted component in the Galactic discs*, Q. F. Xing. & G. Zhao (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`2018MNRAS.476.5388X <http://adsabs.harvard.edu/abs/2018MNRAS.476.5388X>`_)

#. *The GALAH survey: An abundance, age, and kinematic inventory of the solar neighbourhood made with TGAS*, S. Buder, K. Lind, M. K. Ness, et al. (2018) *Astron. & Astrophys.*, submitted (`arXiv/1804.05869 <http://arxiv.org/abs/1804.05869>`_)

#. *The GALAH survey: Co-orbiting stars and chemical tagging*, Jeffrey D. Simpson, Sarah L. Martell, Gary Da Costa, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1804.05894 <http://arxiv.org/abs/1804.05894>`_)

#. *Correlations between age, kinematics, and chemistry as seen by the RAVE survey*, Jennifer Wojno, Georges Kordopatis, Matthias Steinmetz, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1804.06379 <http://arxiv.org/abs/1804.06379>`_)

#. *On the kinematics of a runaway Be star population*, Douglas Boubert & N. Wyn Evans (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1804.05849 <http://arxiv.org/abs/1804.05849>`_)

#. *Unbiased TGAS×LAMOST distances and the role of binarity*, Johanna Coronado, Hans-Walter Rix, & Wilma H. Trick (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1804.07760 <http://arxiv.org/abs/1804.07760>`_)

#. *The Origin of the 300 km/s Stream Near Segue 1*, Sal Wanying Fu, Joshua D. Simon, Matthew Shetrone, et al. (2018) *Astrophys. J.*, submitted (`arXiv/1804.08622 <http://arxiv.org/abs/1804.08622>`_)

#. *Anatomy of the hyper-runaway star LP 40-365 with Gaia*, R. Raddi, M. A. Hollands, B. T. Gaensicke, D. M. Townsley, J. J. Hermes, N. P. Gentile Fusillo, & D. Koester (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1804.09677 <http://arxiv.org/abs/1804.09677>`_)

#. *Revisiting hypervelocity stars after Gaia DR2*, Douglas Boubert, James Guillochon, Keith Hawkins, Idan Ginsburg, & N. Wyn Evans (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1804.10179 <http://arxiv.org/abs/1804.10179>`_)

#. *Gaia Proper Motions and Orbits of the Ultra-Faint Milky Way Satellites*, Joshua D. Simon (2018) *Astrophys. J.*, submitted (`arXiv/1804.10230 <http://arxiv.org/abs/1804.10230>`_)

#. *Three Hypervelocity White Dwarfs in Gaia DR2: Evidence for Dynamically Driven Double-Degenerate Double-Detonation Type Ia Supernovae*, Ken J. Shen, Douglas Boubert, Boris T. Gänsicke, et al. (2018) *Astrophys. J.*, submitted (`arXiv/1804.11163 <http://arxiv.org/abs/1804.11163>`_)

#. *Gaia DR2 Proper Motions of Dwarf Galaxies within 420 kpc: Orbits, Milky Way Mass, Tidal Influences, Planar Alignments, and Group Infall*, T. K. Fritz, G. Battaglia, M. S. Pawlowski, N. Kallivayalil, R. van der Marel, T. S. Sohn, C. Brook, & G. Besla (2018) *Astron. & Astrophys.*, submitted (`arXiv/1805.00908 <http://arxiv.org/abs/1805.00908>`_)

#. *The Lives of Stars: Insights From the TGAS-RAVE-LAMOST Dataset*, John J. Vickers & Martin C. Smith (2018) *Astrophys. J.*, in press (`arXiv/1805.02332 <http://arxiv.org/abs/1805.02332>`_)

#. *High precision pulsar timing and spin frequency second derivatives*, X. J. Liu, C. G. Bassa, & B. W. Stappers (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1805.02892 <http://arxiv.org/abs/1805.02892>`_)
#. *The Galactic Disc in Action Space as seen by Gaia DR2*, Wilma H. Trick, Johanna Coronado, Hans-Walter Rix (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1805.03653 <http://arxiv.org/abs/1805.03653>`_):
     Uses ``galpy.actionAngle.actionAngleStaeckel`` to compute the actions in ``galpy.potential.MWPotential2014`` of stars in the extended solar neighborhood in *Gaia* DR2.
#. *Tidal ribbons*, Walter Dehnen & Hasanuddin (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1805.08481 <http://arxiv.org/abs/1805.08481>`_):
     Employs ``galpy.actionAngle.actionAngleStaeckel`` to compute the actions and frequencies of a disrupted satellite on an orbit close to the Galactic disk, to investigate the difference between the debris from such an orbit and from an orbit well within the halo.

#. *Apocenter Pile-Up: Origin of the Stellar Halo Density Break*, Alis J. Deason, Vasily Belokurov, Sergey E. Koposov, & Lachlan Lancaster (2018) *Astrophys. J. Lett.*, submitted (`arXiv/1805.10288 <http://arxiv.org/abs/1805.10288>`_)

#. *Bootes III is a disrupting dwarf galaxy associated with the Styx stellar stream*, Jeffrey L. Carlin & David J. Sand (2018) *Astrophys. J.*, submitted (`arXiv/1805.11624 <http://arxiv.org/abs/1805.11624>`_)

#. *Proper motions of Milky Way Ultra-Faint satellites with Gaia DR2 × DES DR1*, Andrew B. Pace & Ting S. Li (2018) *Astrophys. J.*, submitted (`arXiv/1806.02345 <http://arxiv.org/abs/1806.02345>`_)

#. *Transient spiral structure and the disc velocity substructure in Gaia DR2*, Jason A. S. Hunt, Jack Hong, Jo Bovy, Daisuke Kawata, Robert J. J. Grand (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1806.02832 <http://arxiv.org/abs/1806.02832>`_):
   	      Builds a simple model of co-rotating, transient spiral structure using the newly implemented ``SpiralArmsPotential`` and the ``CorotatingRotationWrapperPotential`` (and using ``DehnenSmoothWrapperPotential`` to make it transient). Then uses ``galpy.df.evolveddiskdf`` to compute the effect of such spiral structure on the kinematics in the extended solar neighborhood, finding qualitative agreement with the kinematics observed in *Gaia* DR2.

#. *Galactic Archeology with the AEGIS Survey: The Evolution of Carbon and Iron in the Galactic Halo*, Jinmi Yoon, Timothy C. Beers, Sarah Dietz, Young Sun Lee, Vinicius M. Placco, Gary Da Costa, Stefan Keller, Christopher I. Owen, & Mahavir Sharma (2018) *Astrophys. J.*, in press (`arXiv/1806.04738 <http://arxiv.org/abs/1806.04738>`_)

#. *The Formation and Evolution of Galactic Disks with APOGEE and the Gaia Survey*, Chengdong Li, Gang Zhao, Meng Zhai, & Yunpeng Jia (2018) *Astrophys. J.*, in press (`2018ApJ...860...53L <http://adsabs.harvard.edu/abs/2018ApJ...860...53L>`_)

#. *Spectroscopy of Dwarf Stars Around the North Celestial Pole*, Sarunas Mikolaitis, Grazina Tautvaisiene, Arnas Drazdauskas, & Renata Minkeviciute (2018) *Publ. Astron. Soc. Pacific*, in press (`2018PASP..130g4202M <http://adsabs.harvard.edu/abs/2018PASP..130g4202M>`_)

#. *The Study of Galactic Disk Kinematics with SCUSS and SDSS Data*, Xiyan Peng, Zhenyu Wu, Zhaoxiang Qi, Cuihua Du, Jun Ma, Xu Zhou, Yunpeng Jia, & Songhu Wang (2018) *Publ. Astron. Soc. Pacific*, in press (`2018PASP..130g4102P <http://adsabs.harvard.edu/abs/2018PASP..130g4102P>`_)

#. *Common origin for Hercules-Aquila and Virgo Clouds in Gaia DR2*, Iulia T. Simion, Vasily Belokurov, & Sergey E. Koposov (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1807.01335 <http://arxiv.org/abs/1807.01335>`_)

#. *The GALAH Survey and Gaia DR2: (Non)existence of five sparse high-latitude open clusters*, Janez Kos, Gayandhi de Silva, Joss Bland-Hawthorn, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1807.00822 <http://arxiv.org/abs/1807.00822>`_)

#. *Testing the universality of free fall by tracking a pulsar in a stellar triple system*, Anne M. Archibald, Nina V. Gusinskaia, Jason W. T. Hessels, Adam T. Deller, David L. Kaplan, Duncan R. Lorimer, Ryan S. Lynch, Scott M. Ransom, & Ingrid H. Stairs (2018) *Nature*, in press (`arXiv/1807.02059 <http://arxiv.org/abs/1807.02059>`_)

#. *On measuring the Galactic dark matter halo with hypervelocity stars*, O. Contigiani, E. M. Rossi, T. Marchetti (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1807.04468 <http://arxiv.org/abs/1807.04468>`_)

#. *Constraining the Milky Way Halo Potential with the GD-1 stellar stream*, Khyati Malhan & Rodrigo A. Ibata (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1807.05994 <http://arxiv.org/abs/1807.05994>`_)

#. *Interstellar magnetic cannon targeting the Galactic halo : A young bubble at the origin of the Ophiuchus and Lupus molecular complexes*, J.-F. Robitaille, A. M. M. Scaife, E. Carretti, M. Haverkorn, R. M. Crocker, M. J. Kesteven, S. Poppi, & L. Staveley-Smith (2018) *Astron. & Astrophys.*, submitted (`arXiv/1807.04054 <http://arxiv.org/abs/1807.04054>`_)

#. *The GALAH survey: a catalogue of carbon-enhanced stars and CEMP candidates*, Klemen Čotar, Tomaž Zwitter, Janez Kos, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1807.07977 <http://arxiv.org/abs/1807.07977>`_)

#. *Emergence of the Gaia Phase Space Spirals from Bending Waves*, Keir Darling & Lawrence M. Widrow (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1807.11516 <http://arxiv.org/abs/1807.11516>`_)

#. *Pristine Dwarf-Galaxy Survey I: A detailed photometric and spectroscopic study of the very metal-poor Draco II satellite*, Nicolas Longeard, Nicolas Martin, Else Starkenburg, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1807.10655 <http://arxiv.org/abs/1807.10655>`_)

#. *The origin of accreted stellar halo populations in the Milky Way using APOGEE, Gaia, and the EAGLE simulations*, J. Ted Mackereth, Ricardo P. Schiavon, Joel Pfeffer, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1808.00968 <http://arxiv.org/abs/1808.00968>`_)

#. *The vertical motion history of disk stars throughout the Galaxy*, Yuan-Sen Ting & Hans-Walter Rix (2018) *Astrophys. J.*, submitted (`arXiv/1808.03278 <http://arxiv.org/abs/1808.03278>`_)

#. *A kinematical age for the interstellar object 1I/'Oumuamua*, F. Almeida-Fernandes & H. J. Rocha-Pinto (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1808.03637 <http://arxiv.org/abs/1808.03637>`_)

#. *Masses and ages for metal-poor stars: a pilot program combining asteroseismology and high-resolution spectroscopic follow-up of RAVE halo stars*, M. Valentini, C. Chiappini, D. Bossini, et al. (2018) *Astron. & Astrophys.*, submitted (`arXiv/1808.08569 <http://arxiv.org/abs/1808.08569>`_)

#. *On the radial metallicity gradient and radial migration effect of the Galactic disk*, Yunpeng Jia, Yuqin Chen, Gang Zhao, Xiangxiang Xue, Jingkun Zhao, Chengqun Yang, & Chengdong Li (2018) *Astrophys. J.*, in press (`arXiv/1808.05386 <http://arxiv.org/abs/1808.05386>`_)

#. *Rediscovering the Tidal Tails of NGC 288 with Gaia DR2*, Shaziana Kaderali, Jason A. S. Hunt, Jeremy J. Webb, Natalie Price-Jones, & Raymond Carlberg (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1809.04108 <http://arxiv.org/abs/1809.04108>`_)

#. *A Catalog of 10,000 Very Metal-poor Stars from LAMOST DR3*, Haining Li, Kefeng Tan, & Gang Zhao (2018) *Astrophys. J. Supp.*, in press (`ApJS <https://doi.org/10.3847/1538-4365/aada4a>`_)

#. *The AMBRE Project: searching for the closest solar siblings*, V. Adibekyan, P. de Laverny, A. Recio-Blanco, et al. (2018) *Astron. & Astrophys.*, in press (`arXiv/1810.01813 <http://arxiv.org/abs/1810.01813>`_)

#. *SB 796: a high-velocity RRc star*, Roy Gomel, Sahar Shahaf, Tsevi Mazeh, Simchon Faigler, Lisa A. Crause, Ramotholo Sefako, Damien Segransan, Pierre F.L. Maxted, & Igor Soszynski (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1810.09501 <http://arxiv.org/abs/1810.09501>`_)

#. *An Ultra Metal-poor Star Near the Hydrogen-burning Limit*, Kevin C. Schlaufman, Ian B. Thompson, & Andrew R. Casey (2018) *Astrophys. J.*, in press (`arXiv/1811.00549 <http://arxiv.org/abs/1811.00549>`_)

#. *Ages of radio pulsar: long-term magnetic field evolution*, Andrei P. Igoshev (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1810.12922 <http://arxiv.org/abs/1810.12922>`_)

#. *Tracing the formation of the Milky Way through ultra metal-poor stars*, Federico Sestito, Nicolas Longeard, Nicolas F. Martin, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1811.03099 <http://arxiv.org/abs/1811.03099>`_)

#. *GD-1: the relic of an old metal-poor globular cluster*, Guang-Wei Li, Brian Yanny, & Yue Wu (2018) *Astrophys. J.*, submitted (`arXiv/1811.06427 <http://arxiv.org/abs/1811.06427>`_)

#. *Subhalo destruction in the Apostle and Auriga simulations*, Jack Richings, Carlos Frenk, Adrian Jenkins, Andrew Robertson (2018) *Mon. Not. Roy. Astron. Soc.*, to be submitted (`arXiv/1811.12437 <http://arxiv.org/abs/1811.12437>`_)

#. *Searching for the GD-1 Stream Progenitor in Gaia DR2 with Direct N-body Simulations*, Jeremy J. Webb & Jo Bovy (2018) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1811.07022 <http://arxiv.org/abs/1811.07022>`_)

#. *The Origins of Young Stars in the Direction of the Leading Arm of the Magellanic Stream: Abundances, Kinematics, and Orbits*, Lan Zhang, Dana I. Casetti-Dinescu, Christian Moni Bidin, Rene A. Mendez, Terrence M. Girard, Katherine Vieira, Vladimir I. Korchagin, William F. van Altena, & Gang Zhao (2018) *Astrophys. J.*, in press (`arXiv/1812.00198 <http://arxiv.org/abs/1812.00198>`_)

#. *The UTMOST pulsar timing programme I: overview and first results*, F. Jankowski, M. Bailes, W. van Straten, et al. (2018) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1812.04038 <http://arxiv.org/abs/1812.04038>`_)

#. *Dynamical Histories of the Crater II and Hercules Dwarf Galaxies*, Sal Wanying Fu, Joshua D. Simon, Alex G. Alarcon Jara (2019) *Astrophys. J.*, submitted (`arXiv/1901.00594 <http://arxiv.org/abs/1901.00594>`_)

#. *Dynamical heating across the Milky Way disc using APOGEE and Gaia*, J. Ted Mackereth, Jo Bovy, Henry W. Leung, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1901.04502 <http://arxiv.org/abs/1901.04502>`_)

#. *Identifying Sagittarius Stream Stars By Their APOGEE Chemical Abundance Signatures*, Sten Hasselquist, Jeffrey L. Carlin, Jon A. Holtzman, et al. (2019) *Astrophys. J.*, in press (`arXiv/1901.04559 <http://arxiv.org/abs/1901.04559>`_)

#. *The GALAH Survey: Chemodynamics of the Solar Neighbourhood*, Michael R. Hayden, Joss Bland-Hawthorn, Sanjib Sharma, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1901.07565 <http://arxiv.org/abs/1901.07565>`_)

#. *The Pristine Dwarf-Galaxy survey - II. In-depth observational study of the faint Milky Way satellite Sagittarius II*, Nicolas Longeard, Nicolas Martin, Else Starkenburg, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.02780 <http://arxiv.org/abs/1902.02780>`_)

#. *Secular dynamics of binaries in stellar clusters I: general formulation and dependence on cluster potential*, Chris Hamilton & Roman R. Rafikov (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.01344 <http://arxiv.org/abs/1902.01344>`_)

#. *Secular dynamics of binaries in stellar clusters II: dynamical evolution*, Chris Hamilton & Roman R. Rafikov (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.01345 <http://arxiv.org/abs/1902.01345>`_)

#. *Discovery of Tidal Tails in Disrupting Open Clusters: Coma Berenices and a Neighbor Stellar Group*, Shih-Yun Tang, Xiaoying Pang, Zhen Yuan, W. P. Chen, Jongsuk Hong, Bertrand Goldman, Andreas Just, Bekdaulet Shukirgaliyev, & Chien-Cheng Lin (2019) *Astrophys. J.*, submitted (`arXiv/1902.01404 <http://arxiv.org/abs/1902.01404>`_)

#. *A class of partly burnt runaway stellar remnants from peculiar thermonuclear supernovae*, R. Raddi, M. A. Hollands, D. Koester, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.05061 <http://arxiv.org/abs/1902.05061>`_)

#. *Extended stellar systems in the solar neighborhood - III. Like ships in the night: the Coma Berenices neighbor moving group*, Verena Fürnkranz, Stefan Meingast, João Alves (2019) *Astron. & Astrophys.*, submitted (`arXiv/1902.07216 <http://arxiv.org/abs/1902.07216>`_)

#. *Chronostar: a novel Bayesian method for kinematic age determination. I. Derivation and application to the β Pictoris Moving Group*, Timothy D. Crundall, Michael J. Ireland, Mark R. Krumholz, Christoph Federrath, Maruša Žerjal, Jonah T. Hansen (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.07732 <http://arxiv.org/abs/1902.07732>`_)

#. *The GALAH survey and Gaia DR2: Linking ridges, arches and vertical waves in the kinematics of the Milky Way*, Shourya Khanna, Sanjib Sharma, Thor Tepper-Garcia, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1902.10113 <http://arxiv.org/abs/1902.10113>`_)

#. *A Near-coplanar Stellar Flyby of the Planet Host Star HD 106906*, Robert J. De Rosa & Paul Kalas (2019) *Astron. J.*, in press (`arXiv/1902.10220 <http://arxiv.org/abs/1902.10220>`_)

#. *High- and Low-α Disk Stars Separate Dynamically at all Ages*, Suroor S Gandhi & Melissa K Ness (2019) *Astrophys. J.*, submitted (`arXiv/1903.04030 <http://arxiv.org/abs/1903.04030>`_)

#. *The Journey Counts: The Importance of Including Orbits when Simulating Ram Pressure Stripping*, Stephanie Tonnesen (2019) *Astrophys. J.*, in press (`arXiv/1903.08178 <http://arxiv.org/abs/1903.08178>`_)

#. *The moving groups as the origin of the vertical phase space spiral*, Tatiana A. Michtchenko, Douglas A. Barros, Angeles Pérez-Villegas, & Jacques R. D. Lépine (2019) *Astrophys. J.*, in press (`arXiv/1903.08325 <http://arxiv.org/abs/1903.08325>`_)

#. *Origin of the excess of high-energy retrograde stars in the Galactic halo*, Tadafumi Matsuno, Wako Aoki, & Takuma Suda (2019) *Astrophys. J. Lett.*, in press (`arXiv/1903.09456 <http://arxiv.org/abs/1903.09456>`_)

#. *Gaia DR2 orbital properties for field stars with globular cluster-like CN band strengths*, A. Savino & L. Posti (2019) *Astron. & Astrophys. Lett.*, in press (`arXiv/1904.01021 <http://arxiv.org/abs/1904.01021>`_)

#. *Dissecting the Phase Space Snail Shell*, Zhao-Yu Li & Juntai Shen (2019) *Astrophys. J.*, submitted (`arXiv/1904.03314 <http://arxiv.org/abs/1904.03314>`_)

#. *Exploring the age dependent properties of M and L dwarfs using Gaia and SDSS*, Rocio Kiman, Sarah J. Schmidt, Ruth Angus, Kelle L. Cruz, Jacqueline K. Faherty, Emily Rice (2019) *Astron. J.*, in press (`arXiv/1904.05911 <http://arxiv.org/abs/1904.05911>`_)

#. *Signatures of resonance and phase mixing in the Galactic disc*, Jason A. S. Hunt, Mathew W. Bub, Jo Bovy, J. Ted Mackereth, Wilma H. Trick, & Daisuke Kawata (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1904.10968 <http://arxiv.org/abs/1904.10968>`_)

#. *Globular cluster candidates in the Galactic bulge: Gaia and VVV view of the latest discoveries*, F. Gran, M. Zoccali, R. Contreras Ramos, E. Valenti, A. Rojas-Arriagada, J. A. Carballo-Bello, J. Alonso-García, D. Minniti, M. Rejkuba, & F. Surot (2019) *Astron. & Astrophys.*, in press (`arXiv/1904.10872 <http://arxiv.org/abs/1904.10872>`_)

#. *The effect of tides on the Sculptor dwarf spheroidal galaxy*, G. Iorio, C. Nipoti, G. Battaglia, & A. Sollima (2019) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1904.10461 <http://arxiv.org/abs/1904.10461>`_)

#. *Correcting the measured values of the rate of change of the spin and orbital periods of rotation powered pulsars*, Dhruv Pathak & Manjari Bagchi (2019) (`arXiv/1905.01159 <http://arxiv.org/abs/1905.01159>`_)

#. *Are the double-mode bulge RR Lyrae stars with identical period-ratios the relic of a disrupted stellar system?*, Andrea Kunder, Alex Tilton, Dylon Maertens, Jonathan Ogata, David Nataf, R. Michael Rich, Christian I. Johnson, Christina Gilligan, & Brian Chaboyer (2019) *Astrophys. J. Lett.*, in press (`arXiv/1905.03256 <http://arxiv.org/abs/1905.03256>`_)

#. *Evidence for the accretion origin of halo stars with an extreme r-process enhancement*, Qian-Fan Xing, Gang Zhao, Wako Aoki, Satoshi Honda, Hai-Ning Li, Miho N. Ishigaki, & Tadafumi Matsuno (2019) *Nature Astron.*, in press (`arXiv/1905.04141 <http://arxiv.org/abs/1905.04141>`_)

#. *The influence of dark matter halo on the stellar stream asymmetry via dynamical friction*, Rain Kipper, Peeter Tenjes, Gert Hutsi, Taavi Tuvikene, & Elmo Tempel (2019) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1905.05553 <http://arxiv.org/abs/1905.05553>`_)

#. *Origin and Status of Low-Mass Candidate Hypervelocity Stars*, Bum-Suk Yeom, Young Sun Lee, Jae-Rim Koo, Timothy C. Beers, & Young Kwang Kim (2019) *J. Kor. Astron. Soc.*, in press (`arXiv/1905.07879 <http://arxiv.org/abs/1905.07879>`_)

#. *Life in the fast lane: a direct view of the dynamics, formation, and evolution of the Milky Way's bar*, Jo Bovy, Henry W. Leung, Jason A. S. Hunt, J. Ted Mackereth, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1905.11404 <http://arxiv.org/abs/1905.11404>`_)

#. *On the Oosterhoff dichotomy in the Galactic bulge: II. kinematical distribution*, Z. Prudil, I. Dékany, E. K. Grebel, M. Catelan, M. Skarka, & R. Smolec (2019) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1905.11870 <http://arxiv.org/abs/1905.11870>`_)

#. *The Ophiuchus stream progenitor: a new type of globular cluster and its possible Sagittarius connection*, James M. M. Lane, Julio F. Navarro, Azadeh Fattahi, Kyle A. Oman, & Jo Bovy (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1905.12633 <http://arxiv.org/abs/1905.12633>`_)

#. *Identifying resonances of the Galactic bar in Gaia DR2: Clues from action space*, Wilma H. Trick, Francesca Fragkoudi, Jason A. S. Hunt, J. Ted Mackereth, & Simon D. M. White (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1906.04786 <http://arxiv.org/abs/1906.04786>`_)

#. *The Pristine survey - V. A bright star sample observed with SOPHIE*, P. Bonifacio, E. Caffau, F. Sestito, et al. (2019) *Mon. Not. Roy. Astron. Soc.*, in press (`2019MNRAS.487.3797B <http://adsabs.harvard.edu/abs/2019MNRAS.487.3797B>`_)

#. *Metal-poor Stars Observed with the Automated Planet Finder Telescope. II. Chemodynamical Analysis of Six Low-Metallicity Stars in the Halo System of the Milky Way*, Mohammad K. Mardini, Vinicius M. Placco, Ali Taani, Haining Li, & Gang Zhao (2019) *Astrophys. J.*, submitted (`arXiv/1906.08439 <http://arxiv.org/abs/1906.08439>`_)

#. *Applying Liouville's Theorem to Gaia Data*, Matthew R. Buckley, David W. Hogg, & Adrian M. Price-Whelan (2019) *Phys. Rev. D.*, submitted (`arXiv/1907.00987 <http://arxiv.org/abs/1907.00987>`_)

#. *Untangling the Galaxy I: Local Structure and Star Formation History of the Milky Way*, Marina Kounkel & Kevin Covey (2019) *Astron. J.*, in press (`arXiv/1907.07709 <http://arxiv.org/abs/1907.07709>`_)

#. *Constraining churning and blurring in the Milky Way using large spectroscopic surveys -- an exploratory study*, Sofia Feltzing, J. Bradley Bowers, & Oscar Agertz (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1907.08011 <http://arxiv.org/abs/1907.08011>`_)

#. *Proper Motions of Stellar Streams Discovered in the Dark Energy Survey*, N. Shipp, T. S. Li, A. B. Pace, et al. (2019) *Astrophys. J.*, submitted (`arXiv/1907.09488 <http://arxiv.org/abs/1907.09488>`_)

#. *High-resolution spectroscopic study of dwarf stars in the northern sky: Na to Zn abundances in two fields with radii of 20 degrees*, Šarūnas Mikolaitis, Arnas Drazdauskas, Renata Minkevičiūtė, et al. (2019) *Astron. & Astrophys.*, submitted (`arXiv/1907.09157 <http://arxiv.org/abs/1907.09157>`_)

#. *When Cold Radial Migration is Hot: Constraints from Resonant Overlap*, Kathryne J. Daniel, David A. Schaffner, Fiona McCluskey, Codie Fiedler Kawaguchi, & Sarah Loebman (2019) *Astrophys. J.*, submitted (`arXiv/1907.10100 <http://arxiv.org/abs/1907.10100>`_)

#. *In the Galactic disk, stellar [Fe/H] and age predict orbits and precise [X/Fe]*, Melissa K. Ness, Kathryn V. Johnston, Kirsten Blancato, Hans-Walter Rix, Angus Beane, Jonathan C. Bird, & Keith Hawkins (2019) *Astrophys. J.*, submitted (`arXiv/1907.10606 <http://arxiv.org/abs/1907.10606>`_)

#. *General relativistic orbital decay in a seven-minute-orbital-period eclipsing binary system*, Kevin B. Burdge, Michael W. Coughlin, Jim Fuller, et al. (2019) *Nature*, in press (`arXiv/1907.11291 <http://arxiv.org/abs/1907.11291>`_)

#. *Modelling the Effects of Dark Matter Substructure on Globular Cluster Evolution with the Tidal Approximation*, Jeremy J. Webb, Jo Bovy, Raymond G. Carlberg, & Mark Gieles (2019) *Mon. Not. Roy. Astron. Soc.*, in press (`arXiv/1907.13132 <http://arxiv.org/abs/1907.13132>`_)

#. *Kinematic study of the association Cyg OB3 with Gaia DR2*, Anjali Rao, Poshak Gandhi, Christian Knigge, John A. Paice, Nathan W. C. Leigh, & Douglas Boubert (2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1908.00810 <http://arxiv.org/abs/1908.00810>`_)

#. *Gravitational Potential from small-scale clustering in action space: Application to Gaia DR2*, T. Yang, S. S. Boruah, & N. Afshordi(2019) *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1908.02336 <http://arxiv.org/abs/1908.02336>`_)

#. *The Intrinsic Scatter of the Radial Acceleration Relation*, Connor Stone & Stephane Courteau (2019) *Astrophys. J.*, in press (`arXiv/1908.06105 <http://arxiv.org/abs/1908.06105>`_)

#. *Radial Velocity Discovery of an Eccentric Jovian World Orbiting at 18 au*, Sarah Blunt, Michael Endl, Lauren M. Weiss, et al. (2019) *Astron. J.*, in press (`arXiv/1908.09925 <http://arxiv.org/abs/1908.09925>`_)

#. *CCD UBVRI photometry of the open cluster Berkeley 8*, Hikmet Çakmak, Raúl Michel, Yüksel Karataş (2019) *Turkish J. Phys.*, in press (`arXiv/1908.05479 <http://arxiv.org/abs/1908.05479>`_)

#. *The kinematical and space structures of IC 2391 open cluster and moving group with Gaia-DR2*, E. S. Postnikova, W. H. Elsanhoury, Devesh P. Sariya, N. V. Chupina, S. V. Vereshchagin, Ing-Guey Jiang (2019) *Research in Astron. Astrophys.*, submitted (`arXiv/1908.10094 <http://arxiv.org/abs/1908.10094>`_)

#. *Flares of accretion activity of the 20 Myr old UXOR RZ Psc*, I.S. Potravnov, V.P. Grinin, N.A. Serebriakova (2019) *Astron. & Astrophys.*, submitted (`arXiv/1908.08673 <http://arxiv.org/abs/1908.08673>`_)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

