.. galpy documentation master file, created by
   sphinx-quickstart on Sun Jul 11 15:58:27 2010.

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

.. _try_galpy:

Try ``galpy``
-------------

Give ``galpy`` a try in the interactive ``IPython``-like shell below!

.. raw:: html

   <div class="row">
      <div class="column">
        <h3>Plot the rotation curve of the Milky Way</h4>
        <div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">galpy.potential</span> <span class="kn">import</span> <span class="p">(</span><span class="n">plotRotcurve</span><span class="p">,</span>
   <span class="go">        MWPotential2014 as mwp14)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">plotRotcurve</span><span class="p">(</span><span class="n">mwp14</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">plotRotcurve</span><span class="p">(</span><span class="n">mwp14</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Bulge&#39;</span><span class="p">,</span><span class="n">overplot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">plotRotcurve</span><span class="p">(</span><span class="n">mwp14</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Disk&#39;</span><span class="p">,</span><span class="n">overplot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">plotRotcurve</span><span class="p">(</span><span class="n">mwp14</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Halo&#39;</span><span class="p">,</span><span class="n">overplot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span></pre></div></div>
        <h3>or integrate the orbit of MW satellites</h4>
        <div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">galpy.orbit</span> <span class="kn">import</span> <span class="n">Orbit</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">galpy.potential</span> <span class="kn">import</span> <span class="n">MWPotential2014</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">numpy</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">ts</span><span class="o">=</span> <span class="n">numpy</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mf">0.</span><span class="p">,</span><span class="mf">5.</span><span class="p">,</span><span class="mi">2001</span><span class="p">)</span><span class="o">*</span><span class="n">u</span><span class="o">.</span><span class="n">Gyr</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">o</span><span class="o">=</span> <span class="n">Orbit</span><span class="o">.</span><span class="n">from_name</span><span class="p">(</span><span class="s1">&#39;MW satellite galaxies&#39;</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">o</span><span class="o">.</span><span class="n">integrate</span><span class="p">(</span><span class="n">ts</span><span class="p">,</span><span class="n">MWPotential2014</span><span class="p">)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">o</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">xrange</span><span class="o">=</span><span class="p">[</span><span class="mf">0.</span><span class="p">,</span><span class="mf">100.</span><span class="p">],</span><span class="n">yrange</span><span class="o">=</span><span class="p">[</span><span class="o">-</span><span class="mf">100.</span><span class="p">,</span><span class="mf">100.</span><span class="p">])</span>
   <span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">o</span><span class="p">[</span><span class="n">o</span><span class="o">.</span><span class="n">name</span><span class="o">==</span><span class="s1">&#39;LMC&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">c</span><span class="o">=</span><span class="s1">&#39;r&#39;</span><span class="p">,</span><span class="n">lw</span><span class="o">=</span><span class="mf">5.</span><span class="p">,</span><span class="n">overplot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span></pre></div></div>
    <h4>or calculate the Sun's orbital actions, frequencies, and angles</h4>
    <div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">galpy.orbit</span> <span class="kn">import</span> <span class="n">Orbit</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">galpy.potential</span> <span class="kn">import</span> <span class="p">(</span>
   <span class="go">        MWPotential2014 as mwp14)</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="n">o</span><span class="o">=</span> <span class="n">Orbit</span><span class="p">()</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">o</span><span class="o">.</span><span class="n">jr</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">),</span><span class="n">o</span><span class="o">.</span><span class="n">Lz</span><span class="p">(),</span><span class="n">o</span><span class="o">.</span><span class="n">jz</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">))</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">o</span><span class="o">.</span><span class="n">Or</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">),</span><span class="n">o</span><span class="o">.</span><span class="n">Op</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">),</span><span class="n">o</span><span class="o">.</span><span class="n">Oz</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">))</span>
   <span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">o</span><span class="o">.</span><span class="n">wr</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">),</span><span class="n">o</span><span class="o">.</span><span class="n">wp</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">),</span><span class="n">o</span><span class="o">.</span><span class="n">wz</span><span class="p">(</span><span class="n">pot</span><span class="o">=</span><span class="n">mwp14</span><span class="p">))</span></pre></div></div>
           <h3>and much more... Start your journey below</h4>
      </div>
      <div class="column" style="padding-top: 40px;">
         <div id="activate-try-galpy">
            <input class='activatetrygalpybutton' id='activate-try-galpy-button' type="button" value="Activate interactive session">
         </div>
        <!-- REPL iframe inserted by try-galpy.js -->
      <br/>
      <p style="font-size: small;">(This interactive shell runs using the <a href="https://pyodide.org/en/stable/" target="_blank"><span class="courier-code">pyodide</span></a>
         Python kernel, a version of Python that runs in your browser. <span class="courier-code">galpy</span> runs in your browser at compiled-C-like speed! Please open an <a href="https://github.com/jobovy/galpy/issues" target="_blank">Issue</a> for any problems that you find with the interactive session.)</p>
      </div>
    </div>

Quick-start guide
-----------------

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install galpy on Linux, Mac, or Windows via pip or conda,
      including optional C extensions and configuration.

   .. grid-item-card:: What's New
      :link: whatsnew
      :link-type: doc

      Release notes and changelog for all galpy versions.

   .. grid-item-card:: Rotation Curves, Units, and First Orbits
      :img-top: images/tutorials/getting_started_rotation_curves_and_units.png
      :link: tutorials/getting_started/rotation_curves_and_units
      :link-type: doc

      Set up basic potentials, plot rotation curves, understand
      galpy's natural and physical unit systems, and integrate
      a simple orbit.

Potentials
^^^^^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Introduction to Potentials
      :img-top: images/tutorials/potentials_introduction.png
      :link: tutorials/potentials/introduction
      :link-type: doc

      Evaluate potentials, forces, and densities. Combine potentials,
      plot rotation curves and potential contours, compute orbital
      frequencies and Lindblad resonances.

   .. grid-item-card:: Milky Way-like Potentials
      :img-top: images/tutorials/potentials_milky_way_potentials.png
      :link: tutorials/potentials/milky_way_potentials
      :link-type: doc

      Use built-in Milky Way potentials: MWPotential2014, McMillan17,
      Irrgang13, Cautun20, DehnenBinney98. Add a central black hole
      or dynamical friction.

   .. grid-item-card:: Potential Wrappers
      :img-top: images/tutorials/potentials_wrappers.png
      :link: tutorials/potentials/wrappers
      :link-type: doc

      Modify potentials with time-dependent amplitudes, solid-body
      rotation, tilting, and offsetting using wrapper classes.

   .. grid-item-card:: SCF and Multipole Expansions
      :img-top: images/tutorials/potentials_scf_and_multipole.png
      :link: tutorials/potentials/scf_and_multipole
      :link-type: doc

      Represent arbitrary density distributions using basis-function
      (SCF) or multipole expansions, including time-dependent and
      disky potentials.

   .. grid-item-card:: Dissipative Forces
      :img-top: images/tutorials/potentials_dissipative.png
      :link: tutorials/potentials/dissipative
      :link-type: doc

      Use velocity-dependent forces like Chandrasekhar dynamical
      friction and non-inertial frame forces for orbit integration.
      Includes the Schwarzschild precession of S2 around Sgr A*.

   .. grid-item-card:: N-body Simulation Potentials
      :img-top: images/tutorials/potentials_nbody_snapshots.png
      :link: tutorials/potentials/nbody_snapshots
      :link-type: doc

      Use frozen N-body simulation potentials in galpy
      with pynbody for orbits and analysis.

   .. grid-item-card:: Using Potentials in Other Codes
      :img-top: images/potential-amuse-example.png
      :link: tutorials/potentials/other_codes
      :link-type: doc

      Export galpy potentials to NEMO, AMUSE, AGAMA, and gala
      for use in other simulation frameworks.

Orbits
^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Orbit Initialization
      :img-top: images/tutorials/orbits_initialization.png
      :link: tutorials/orbits/initialization
      :link-type: doc

      Initialize orbits in cylindrical coordinates, with physical
      units, from observed coordinates (RA/Dec, Galactic), or from
      astropy SkyCoord objects.

   .. grid-item-card:: Multiple Orbits
      :img-top: images/tutorials/orbits_multiple_orbits.png
      :link: tutorials/orbits/multiple_orbits
      :link-type: doc

      Work with multiple orbits at once: array and SkyCoord
      initialization, slicing, reshaping, and parallel integration.

   .. grid-item-card:: Orbits of Known Objects
      :img-top: images/tutorials/orbits_known_objects.png
      :link: tutorials/orbits/known_objects
      :link-type: doc

      Initialize orbits from object names using built-in catalogs or
      SIMBAD. Load collections of MW globular clusters, satellite
      galaxies, or the solar system.

   .. grid-item-card:: Integration and Plotting
      :img-top: images/tutorials/orbits_integration_and_plotting.png
      :link: tutorials/orbits/integration_and_plotting
      :link-type: doc

      Integrate orbits, display various projections, access orbital
      quantities, check energy conservation, use non-inertial frames,
      and compute surfaces of section.

   .. grid-item-card:: Fast Orbit Characterization
      :img-top: images/tutorials/orbits_fast_characterization.png
      :link: tutorials/orbits/fast_characterization
      :link-type: doc

      Quickly compute eccentricity, peri/apocenter, and zmax using
      the Staeckel approximation without orbit integration.

   .. grid-item-card:: Orbit Examples
      :img-top: images/tutorials/orbits_examples.png
      :link: tutorials/orbits/examples
      :link-type: doc

      Detailed examples: LMC orbit with dynamical friction,
      barycentric acceleration from the LMC, and more.

Distribution Functions
^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Two-dimensional Disk DFs
      :img-top: images/tutorials/distribution_functions_disk_df_2d.png
      :link: tutorials/distribution_functions/disk_df_2d
      :link-type: doc

      Dehnen, Shu, and Schwarzschild distribution functions for
      razor-thin disks: moments, asymmetric drift, Oort constants,
      sampling, and non-axisymmetric evolution.

   .. grid-item-card:: Three-dimensional Disk DFs
      :img-top: images/tutorials/distribution_functions_disk_df_3d.png
      :link: tutorials/distribution_functions/disk_df_3d
      :link-type: doc

      Quasi-isothermal distribution function for 3D disk populations
      using action-angle variables: moments, velocity ellipsoid tilt,
      and velocity sampling.

   .. grid-item-card:: Spherical DFs
      :img-top: images/tutorials/distribution_functions_spherical_dfs.png
      :link: tutorials/distribution_functions/spherical_dfs
      :link-type: doc

      Isotropic and anisotropic distribution functions for spherical
      systems: Hernquist, NFW, King, Plummer, and Osipkov-Merritt models.

Action-Angle Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Introduction to Action-Angle Coordinates
      :img-top: images/tutorials/action_angle_introduction.png
      :link: tutorials/action_angle/introduction
      :link-type: doc

      Overview of action-angle variables, using the Orbit interface,
      and calculating actions for isochrone and spherical potentials.

   .. grid-item-card:: Staeckel Approximation
      :img-top: images/tutorials/action_angle_staeckel.png
      :link: tutorials/action_angle/staeckel
      :link-type: doc

      The most accurate general method for computing actions in
      axisymmetric potentials, with grid-based interpolation for speed.

   .. grid-item-card:: Adiabatic Approximation
      :img-top: images/tutorials/action_angle_adiabatic.png
      :link: tutorials/action_angle/adiabatic
      :link-type: doc

      Fast action calculation by separating radial and vertical
      motions, best for thin-disk orbits.

   .. grid-item-card:: Orbit Integration-based Method
      :img-top: images/tutorials/action_angle_isochroneapprox.png
      :link: tutorials/action_angle/isochroneapprox
      :link-type: doc

      General action-angle calculation by integrating the orbit and
      using a best-fit isochrone potential.

   .. grid-item-card:: 1D Actions and Inverse Transformations
      :img-top: images/tutorials/action_angle_vertical_and_inverse.png
      :link: tutorials/action_angle/vertical_and_inverse
      :link-type: doc

      One-dimensional action-angle coordinates for vertical
      oscillations and their inverse transformations.

   .. grid-item-card:: Inverse Transformations (TorusMapping)
      :img-top: images/tutorials/action_angle_torus.png
      :link: tutorials/action_angle/torus
      :link-type: doc

      Compute phase-space coordinates from given actions and angles
      using the Torus Mapper.

Tidal Streams
^^^^^^^^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Stream Modeling with streamdf
      :img-top: images/tutorials/streams_streamdf.png
      :link: tutorials/streams/streamdf
      :link-type: doc

      Model tidal streams in action-angle coordinates: predict stream
      tracks, compute densities, sample mock data, and evaluate the
      full phase-space PDF.

   .. grid-item-card:: Particle-spray with streamspraydf
      :img-top: images/tutorials/streams_streamspraydf.png
      :link: tutorials/streams/streamspraydf
      :link-type: doc

      Generate tidal streams using particle-spray techniques with
      chen24spraydf and fardal15spraydf, including progenitor
      self-gravity.

Extending galpy
^^^^^^^^^^^^^^^

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: New Potentials in Python
      :img-top: images/tutorials/extending_new_potential_python.png
      :link: tutorials/extending/new_potential_python
      :link-type: doc

      Define a custom potential class in Python by implementing
      force and potential evaluation methods, with full support for
      orbit integration and physical units.

   .. grid-item-card:: New Potentials in C
      :link: tutorials/extending/new_potential_c
      :link-type: doc

      Add C implementations of potentials for fast orbit integration
      and action-angle calculations.

.. toctree::
   :hidden:

   installation.rst
   whatsnew.rst
   tutorials/getting_started/index
   tutorials/potentials/index
   tutorials/orbits/index
   tutorials/distribution_functions/index
   tutorials/action_angle/index
   tutorials/streams/index
   tutorials/extending/index

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
* ``galpy.df.chen24spraydf``: please cite `Chen et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract>`__ for the method and the ``galpy`` implementation
* ``galpy.df.fardal15spraydf``: please cite `Fardal et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract>`__ for the method and `Qian et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2339Q/abstract>`__ for the ``galpy`` implementation
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
