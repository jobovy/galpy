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
* ``galpy.df.streamspraydf``: please cite `Fardal et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract>`__ for the method and `Qian et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2339Q/abstract>`__ for the ``galpy`` implementation
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

