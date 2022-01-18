.. galpy documentation master file, created by
   sphinx-quickstart on Sun Jul 11 15:58:27 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. ifconfig:: not_on_rtd

   .. WARNING:: You are looking at the rarely updated, GitHub version of this documentation, **please go to** `galpy.readthedocs.io <http://galpy.readthedocs.io>`_ **for the latest documentation**.

Welcome to galpy's documentation
=================================

`galpy <http://www.galpy.org>`__ is a Python package for galactic
dynamics. It supports orbit integration in a variety of potentials,
evaluating and sampling various distribution functions, and the
calculation of action-angle coordinates for all static
potentials. galpy is an `astropy <http://www.astropy.org/>`_
`affiliated package <http://www.astropy.org/affiliated/>`_ and
provides full support for astropy's `Quantity
<http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`_
framework for variables with units.

galpy is developed on GitHub. If you are looking to `report an issue
<https://github.com/jobovy/galpy/issues>`_, `join <https://join.slack.com/t/galpy/shared_invite/zt-p6upr4si-mX7u8MRdtm~3bW7o8NA_Ww>`_ the `galpy slack <https://galpy.slack.com/>`_ community, or for information on how
to `contribute to the code
<https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors>`_,
please head over to `galpy's GitHub page
<https://github.com/jobovy/galpy>`_ for more information.

As a preview of the kinds of things you can do with galpy, here's a
video introducing some of the new features in galpy v1.5:

.. raw:: html

   <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce the new version (v1.5) of galpy!<a href="https://t.co/sVlP7utkc6">https://t.co/sVlP7utkc6</a><br>Many new features, see them all here:<a href="https://t.co/KZ6ZaQCNvv">https://t.co/KZ6ZaQCNvv</a><br>Watch the video below for a quick intro to some of the most exciting new capabilities: <a href="https://t.co/HqUgAeVz24">pic.twitter.com/HqUgAeVz24</a></p>&mdash; Jo Bovy (@jobovy) <a href="https://twitter.com/jobovy/status/1173590198225125376?ref_src=twsrc%5Etfw">September 16, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

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
* ``galpy.potential.ttensor`` and ``galpy.potential.rtide``: please cite `Webb et al. (2019a) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5748W/abstract>`__.

* ``galpy.potential.to_amuse``: please cite `Webb et al. (2019b) <http://arxiv.org/abs/1910.01646>`_.

Please also send me a reference to the paper or send a pull request
including your paper in the list of galpy papers on this page (this
page is at doc/source/index.rst). Thanks!

Papers using galpy
--------------------

.. raw:: html

   <p><code class="docutils literal notranslate"><span class="pre">galpy</span></code> has been used in more than <span id="span-number-of-papers-using-galpy">200</span> scientific publications in the astrophysical literature. Covering topics as diverse as the properties of planetary systems around distant stars, the kinematics of pulsars and stars ejected from the Milky Way by supernova explosions, binary evolution in stellar clusters, chemo-dynamical modeling of stellar populations in the Milky Way, and the dynamics of satellites around external galaxies, <code class="docutils literal notranslate"><span class="pre">galpy</span></code> is widely used throughout astrophysics.<br/><br/>Check out the gallery below for examples:<br/>
   </p>

   <div class="papers-gallery" id="papers-gallery">
     <!-- Populated using Javascript loading of JSON file from galpy.org -->
   </div>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

