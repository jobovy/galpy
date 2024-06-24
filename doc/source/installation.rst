.. _installation:

Installation
==============

.. _tldr_installation:

TL;DR
------

.. tab-set::
    :class: platform-selector-tabset

    .. tab-item:: Linux

        The recommended way to install is using ``pip``::

            python -m pip install --only-binary galpy galpy

        This should install a fully-working version of galpy for Python versions >=3.8. If this fails, please open an `issue <https://github.com/jobovy/galpy/issues/new?assignees=&labels=&template=bug_report.md&title=>`__ on the ``galpy`` GitHub page, making sure to specify your platform and Python version. Then read on at :ref:`detailed_installation` to learn how to install ``galpy`` when the above fails.

    .. tab-item:: Mac

        The recommended way to install is using ``pip``::

            python -m pip install --only-binary galpy galpy

        This should install a fully-working version of galpy for Python versions >=3.8. If this fails, please open an `issue <https://github.com/jobovy/galpy/issues/new?assignees=&labels=&template=bug_report.md&title=>`__ on the ``galpy`` GitHub page, making sure to specify your platform and Python version. Then read on at :ref:`detailed_installation` to learn how to install ``galpy`` when the above fails.

    .. tab-item:: Windows

        The recommended way to install is using ``conda``::

            conda install -c conda-forge gsl galpy

        Note that on Windows it is necessary to explicitly install the GNU Scientific Library (GSL) in this way.

For more info on installation options, jump to the detailed instructions below: :ref:`detailed_installation`


.. _deps_installation:

Dependencies
------------

galpy requires the ``packaging``, ``numpy``, ``scipy``, and ``matplotlib`` packages;
these must be installed or the code will not be able to be imported.
All installation methods described on this page will automatically install
these required dependencies.

Optional dependencies are:

  * ``astropy`` for `Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__ support (used throughout galpy when installed),
  * ``astroquery`` for the ``Orbit.from_name`` initialization method (to initialize using a celestial object's name),
  * ``tqdm`` for displaying a progress bar for certain operations (e.g., orbit integration of multiple objects at once)
  * ``numexpr`` for plotting arbitrary expressions of ``Orbit`` quantities,
  * ``numba`` for speeding up the evaluation of certain functions when using C orbit integration,
  * ``JAX`` for use of constant-anisotropy DFs in ``galpy.df.constantbetadf``, and
  * `pynbody <https://github.com/pynbody/pynbody>`__ for use of ``SnapshotRZPotential`` and ``InterpSnapshotRZPotential``.

.. _detailed_installation:

Detailed installation instructions
----------------------------------

If you are reading this, either the simple installation instructions at the top of this page did not work, you are trying to install the latest bleeding-edge version, or you are trying to set up a development installation. For info on setting up a development installation, see :ref:`dev_installation`. In this section, we cover how to install the latest release *or* the latest bleeding-edge version for use on different platforms.

.. WARNING::
   When using ``conda`` to install the GSL, make sure to install it from the
   ``conda-forge`` channel using ``conda install -c conda-forge gsl``. If you
   install the GSL from the ``anaconda`` channel, it will often not work with
   ``galpy``. In an ``environment.yml`` file, use ``- conda-forge::gsl``.

.. tab-set::
    :class: platform-selector-tabset

    .. tab-item:: Linux

        As discussed in the :ref:`tldr_installation` section above, the simplest and
        quickest way to install the latest ``galpy`` release on Linux is to use
        ``pip``::

            python -m pip install --only-binary galpy galpy

        Alternatively, you can install both the GSL and ``galpy`` using ``conda``::

            conda install -c conda-forge gsl galpy

        To compile ``galpy`` from source, you will first need to install the GSL. The
        easiest way to do this is using your package manager. On Linux distributions
        with ``apt-get``, do::

            apt-get install libgsl0-dev

        or on distros with ``yum``, do::

            yum install gsl-devel

        Alternatively, you can use ``conda`` to install the GSL and use ``conda`` to
        manage your Python environment. Install the GSL in your preferred environment
        with::

            conda install -c conda-forge gsl

        Once you have installed the GSL, compile ``galpy`` from source using::

            export CFLAGS="$CFLAGS -I`gsl-config --prefix`/include"
            export LDFLAGS="$LDFLAGS -L`gsl-config --prefix`/lib"
            python -m pip install --no-binary galpy galpy

        The commands in this section so far all install the latest release. If you want
        to install the latest bleeding-edge version, you have two options. If the
        installation in the :ref:`tldr_installation` works for you, you can download a
        binary wheel for the latest ``main`` branch version on GitHub, which is available
        `here <http://www.galpy.org.s3-website.us-east-2.amazonaws.com/list.html>`__.
        To install these wheels, download the relevant version for your operating system
        and Python version and do::

            python -m pip install WHEEL_FILE.whl

        These wheels have stable ``...latest...`` names, so you can embed them in
        workflows that should always be using the latest version of ``galpy``
        (e.g., to test your code against the latest development version).

        If this doesn't work, follow the steps above to install the GSL, define the
        relevant environment variables, and then install from source using::

            python -m pip install git+https://github.com/jobovy/galpy.git#egg=galpy

        You can also download the source code or clone the repository, navigate to the
        top-level directory, and install using::

            python -m pip install .

    .. tab-item:: Mac

        As discussed in the :ref:`tldr_installation` section above, the simplest and
        quickest way to install the latest ``galpy`` release on a Mac is to use
        ``pip``::

            python -m pip install --only-binary galpy galpy

        Alternatively, you can install both the GSL and ``galpy`` using ``conda``::

            conda install -c conda-forge gsl galpy

        To compile ``galpy`` from source, you will first need to install the GSL. The
        easiest way to do this is using `Homebrew <http://brew.sh/>`__ as::

            brew install gsl

        Alternatively, you can use ``conda`` to install the GSL and use ``conda`` to
        manage your Python environment. Install the GSL in your preferred environment
        with::

            conda install -c conda-forge gsl

        Once you have installed the GSL, compile ``galpy`` from source using::

            export CFLAGS="$CFLAGS -I`gsl-config --prefix`/include"
            export LDFLAGS="$LDFLAGS -L`gsl-config --prefix`/lib"
            python -m pip install --no-binary galpy galpy

        The commands in this section so far all install the latest release. If you want
        to install the latest bleeding-edge version, you have two options. If the
        installation in the :ref:`tldr_installation` works for you, you can download a
        binary wheel for the latest ``main`` branch version on GitHub, which is
        available `here <http://www.galpy.org.s3-website.us-east-2.amazonaws.com/list.html>`__.
        To install these wheels, download the relevant version for your operating
        system and Python version and do::

            python -m pip install WHEEL_FILE.whl

        These wheels have stable ``...latest...`` names, so you can embed them in
        workflows that should always be using the latest version of ``galpy``
        (e.g., to test your code against the latest development version).

        If this doesn't work, follow the steps above to install the GSL, define the
        relevant environment variables, and then install from source using::

            python -m pip install git+https://github.com/jobovy/galpy.git#egg=galpy

        You can also download the source code or clone the repository, navigate to the
        top-level directory, and install using::

            python -m pip install .

    .. tab-item:: Windows

        As discussed in the :ref:`tldr_installation` section above, the simplest and
        quickest way to install the latest ``galpy`` release on Windows is to use
        ``conda``::

            conda install -c conda-forge gsl galpy

        If you want to install the latest bleeding-edge version, you have to install
        the GSL first as. In an existing ``conda`` environment, do::

            conda install -c conda-forge gsl

        while if you don't want to use ``conda`` to manage your Python environment, you
        can do::

            conda create -n gsl conda-forge::gsl
            conda activate gsl

        Either way, then set the path and relevant environment variables using::

            set PATH=%PATH%;"$CONDA_PREFIX\\Library\\bin"
            set INCLUDE=%CONDA_PREFIX%\Library\include;%INCLUDE%
            set LIB=%CONDA_PREFIX%\Library\lib;%LIB%
            set LIBPATH=%CONDA_PREFIX%\Library\lib;%LIBPATH%

        in the ``CMD`` shell or::

            $env:Path+="$env:CONDA_PREFIX\Library\bin"
            $env:INCLUDE="$env:CONDA_PREFIX\Library\include"
            $env:LIB="$env:CONDA_PREFIX\Library\lib"
            $env:LIBPATH="$env:CONDA_PREFIX\Library\lib"

        if you are using ``PowerShell``. Note that you have to execute these commands
        from the ``conda`` environment such that the ``CONDA_PREFIX`` variable is set.
        To compile with OpenMP on Windows, you have to also install Intel OpenMP via::

            conda install -c anaconda intel-openmp

        Then you can deactivate the conda environment (but you don't have to!).

        With the GSL set up, you can then download a binary wheel for the latest
        ``main`` branch version on GitHub, which is available
        `here <http://www.galpy.org.s3-website.us-east-2.amazonaws.com/list.html>`__.
        To install these wheels, download the relevant version for your operating
        system and Python version and do::

            python -m pip install WHEEL_FILE.whl

        You can also compile from source using::

            python -m pip install git+https://github.com/jobovy/galpy.git#egg=galpy

        or you can download the source code or clone the repository, navigate to the
        top-level directory, and install using::

            python -m pip install .

        Whenever you run ``galpy``, you have to adjust the ``PATH`` variable as above.
        These wheels have stable ``...latest...`` names, so you can embed them in
        workflows that should always be using the latest version of ``galpy``
        (e.g., to test your code against the latest development version).

.. _dev_installation:

Development installation
------------------------

To install ``galpy`` for local development (i.e., you are changing the
``galpy`` source code), first fork the repository on GitHub to your own account
and then clone it to your local machine::

    git clone git@github.com:YOUR_GITHUB_USERNAME/galpy.git

Then, install the GSL as described in the relevant :ref:`detailed_installation`
section above. Once you have installed the GSL, compile ``galpy`` from source::

    export CFLAGS="$CFLAGS -I`gsl-config --prefix`/include"
    export LDFLAGS="$LDFLAGS -L`gsl-config --prefix`/lib"
    python -m pip install -e .

Whenever you change the C code, you have to re-run the last command. Note that
any development should happen on a branch with an informative name.

To test the code locally, install ``pytest``::

    pip install pytest

You might also need to make sure to install the optional dependencies as
discussed :ref:`here <deps_installation>` depending on which tests you want to
run. Running the entire test code takes a long time and isn't recommended (CI
does that). Tests are arranged in files for large chunks of related
functionality and you would typically run a single one of these locally. For
example::

    pytest -vxs tests/test_coords.py

The '-v' flag is for verbose output, the '-x' flag stops after the first failure,
and the '-s' flag prints output from print statements. You can also run a single
test in a file, e.g.,::

    pytest -vxs tests/test_coords.py::test_radec_to_lb_ngp

to just run the ``test_radec_to_lb_ngp`` test. You can also run tests with names
that match a pattern, e.g.,::

    pytest -vxs tests/test_coords.py -k "ngp"

to run all tests in ``test_coords.py`` that have ``ngp`` in their name.

``galpy`` uses `pre-commit <https://pre-commit.com/>`__ to run a number of
pre-commit checks on the code. To install pre-commit, do::

    pip install pre-commit

and then run::

    pre-commit install

to install the pre-commit hooks. These will run automatically whenever you commit.

More esoteric installations
---------------------------

.. _install_tm:

Installing the TorusMapper code
+++++++++++++++++++++++++++++++

.. WARNING::
   The TorusMapper code is *not* part of any of galpy's binary distributions (installed using conda or pip); if you want to gain access to the TorusMapper, you need to install from source as explained in this section and above.

Since v1.2, ``galpy`` contains a basic interface to the TorusMapper
code of `Binney & McMillan (2016)
<http://adsabs.harvard.edu/abs/2016MNRAS.456.1982B>`__. This interface
uses a stripped-down version of the TorusMapper code, that is not
bundled with the galpy code, but kept in a fork of the original
TorusMapper code. Installation of the TorusMapper interface is
therefore only possible when installing from source after downloading
or cloning the galpy code and installing using ``pip install .``.

To install the TorusMapper code, clone the ``galpy`` repository and *before*
running the installation of
galpy, navigate to the top-level galpy directory (which contains the
``setup.py`` file) and do::

	     git clone https://github.com/jobovy/Torus.git galpy/actionAngle/actionAngleTorus_c_ext/torus
	     cd galpy/actionAngle/actionAngleTorus_c_ext/torus
	     git checkout galpy
	     cd -

Then proceed to install galpy using the ``pip install .``
technique or its variants as usual.

.. _install_pyodide:

Using ``galpy`` in web applications
+++++++++++++++++++++++++++++++++++

``galpy`` can be compiled to `WebAssembly <https://webassembly.org/>`__ using the `emscripten <https://emscripten.org/>`__ compiler. In particular, ``galpy`` is part of the `pyodide <https://pyodide.org/en/stable/>`__ Python distribution for the browser, meaning that ``galpy`` can be used on websites without user installation and it still runs at the speed of a compiled language. This powers, for example, the :ref:`Try galpy <try_galpy>` interactive session on this documentation's home page. Thus, it is easy to, e.g., build web-based, interactive galactic-dynamics examples or tutorials without requiring users to install the scientific Python stack and ``galpy`` itself.

``galpy`` is included in versions >0.20 of ``pyodide``, so ``galpy`` can be imported in any web context that uses ``pyodide`` (e.g., `jupyterlite <https://jupyterlite.readthedocs.io/en/latest/>`__ or `pyscript <https://pyscript.net/>`__). Python packages used in ``pyodide`` are compiled to the usual wheels, but for the ``emscripten`` compiler. Such a wheel for the latest development version of ``galpy`` is always available at `galpy-latest-cp310-cp310-emscripten_wasm32.whl <https://www.galpy.org/wheelhouse/galpy-latest-cp310-cp310-emscripten_wasm32.whl>`__ (note that this URL will change for future ``pyodide`` versions, which include ``emscripten`` version numbers in the wheel name). It can be used in ``pyodide`` for example as

>>> import pyodide_js
>>> await pyodide_js.loadPackage(['numpy','scipy','matplotlib','astropy',
        'future','setuptools',
        'https://www.galpy.org/wheelhouse/galpy-latest-cp310-cp310-emscripten_wasm32.whl'])

after which you can ``import galpy`` and do (almost) everything you can in the Python version of ``galpy`` (everything except for querying Simbad using ``Orbit.from_name`` and except for ``Orbit.animate``). Note that depending on your context, you might have to just ``import pyodide`` to get the ``loadPackage`` function.


Installation FAQ
-----------------

I get warnings like "galpyWarning: libgalpy C extension module not loaded, because libgalpy.so image was not found"
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This typically means that the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_) was unavailable during galpy's
installation, causing the C extensions not to be compiled. Most of the
galpy code will still run, but slower because it will run in pure
Python. The code requires GSL versions >= 1.14. If you believe that
the correct GSL version is installed for galpy, check that the library
can be found during installation (see :ref:`below <gsl_cflags>`).

I get warnings like "libgalpy C extension module not loaded, because of error 'dlopen(....../site-packages/libgalpy.cpython-310-darwin.so, 0x0006): Library not loaded: '@rpath/libgsl.25.dylib' etc."
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This happens when ``galpy`` was successfully compiled against the GSL, but the
GSL is not available at runtime. This mainly happens when you installed a binary
package (e.g,, using ``conda`` or a Windows wheel from ``pip``) and you don't
have the GSL or the correct version available locally.

For example, this commonly happens when you have installed the GSL using
``conda`` from the ``anaconda`` channel, which often happens because most people
have the ``defaults`` channel at higher priority than the ``conda-forge`` channel.
Use::

    conda list gsl

to check the channel from which the GSL was installed. If it was not the
``conda-forge`` channel, uninstall the GSL::

    conda uninstall gsl

and re-install from ``conda-forge``::

    conda install -c conda-forge gsl

I get the warning "galpyWarning: libgalpy_actionAngleTorus C extension module not loaded, because libgalpy_actionAngleTorus.so image was not found"
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is typically because the TorusMapper code was not compiled,
because it was unavailable during installation. This code is only
necessary if you want to use
``galpy.actionAngle.actionAngleTorus``. See :ref:`above <install_tm>`
for instructions on how to install the TorusMapper code. Note that in
recent versions of galpy, you should *not* be getting this warning,
unless you set ``verbose=True`` in the :ref:`configuration file
<configfile>`.

.. _gsl_install:

How do I install the GSL?
++++++++++++++++++++++++++

The easiest way to install this is using its Anaconda build::

	conda install -c conda-forge gsl

If you do not want to go that route, on a Mac, the next easiest way to install
the GSL is using `Homebrew <http://brew.sh/>`_ as::

	brew install gsl

You should be able to check your version using (on Mac/Linux)::

   gsl-config --version

On Linux distributions with ``apt-get``, the GSL can be installed using::

   apt-get install libgsl0-dev

or on distros with ``yum``, do::

   yum install gsl-devel

On Windows, using ``conda-forge`` to install the GSL is your best bet, but note
that this doesn't mean that you have to use ``conda`` for the rest of your
Python environment. You can simply use a ``conda`` environment for the GSL,
while using ``pip`` to install ``galpy`` and other packages. However, in that
case, you need to add the relevant ``conda`` environment to your ``PATH``.
So, for example, you can install the GSL as::

    conda create -n gsl conda-forge::gsl
    conda activate gsl

and then set the path using::

    set PATH=%PATH%;"$CONDA_PREFIX\\Library\\bin"

in the ``CMD`` shell or::

    $env:Path+="$env:CONDA_PREFIX\Library\bin"

if you are using ``PowerShell``. Note that you have to execute these commands
from the ``conda`` environment such that the ``CONDA_PREFIX`` variable is set.
You also still have to set the ``INCLUDE``, ``LIB``, and ``LIBPATH`` environment
variables as discussed in :ref:`detailed_installation_win` (also from the conda
environment). Then you can deactivate the conda environment and install
``galpy`` using, e.g., ``pip``. Whenever you run ``galpy``, you have to adjust
the ``PATH`` variable as above.

.. _gsl_cflags:

The ``galpy`` installation fails because of C compilation errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``galpy``'s installation from source can fail due to compilation errors, which
look like::

    error: command 'gcc' failed with exit status 1

or::

	error: command 'clang' failed with exit status 1

or::

	error: command 'cc' failed with exit status 1

This is typically because the compiler cannot locate the GSL header
files or the GSL library. You can tell the installation about where
you've installed the GSL library by defining (for example, when the
GSL was installed under ``/usr``; the ``LD_LIBRARY_PATH`` part of this
may or may not be necessary depending on your system)::

    export CFLAGS=-I/usr/include
    export LDFLAGS=-L/usr/lib
    export LD_LIBRARY_PATH=/usr/lib

or::

	setenv CFLAGS -I/usr/include
	setenv LDFLAGS -L/usr/lib
	setenv LD_LIBRARY_PATH /usr/lib

depending on your shell type (change the actual path to the include
and lib directories that have the gsl directory). If you already have
``CFLAGS``, ``LDFLAGS``, and ``LD_LIBRARY_PATH`` defined you just have
to add the ``'-I/usr/include'``, ``'-L/usr/lib'``, and ``'/usr/lib'`` to
them.

If you are on a Mac or UNIX system (e.g., Linux), you can find the correct
``CFLAGS`` and ``LDFLAGS``/``LD_LIBRARY_path`` entries by doing::

   gsl-config --cflags
   gsl-config --prefix

where you should add ``/lib`` to the output of the latter.

I have defined ``CFLAGS``, ``LDFLAGS``, and ``LD_LIBRARY_PATH``, but the compiler does not seem to include these and still returns with errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This typically happens if you install using ``sudo``, but have defined the
``CFLAGS`` etc. environment variables without using sudo. Try using ``sudo -E``
instead, which propagates your own environment variables to the ``sudo`` user.

I'm having issues with OpenMP
+++++++++++++++++++++++++++++++

galpy uses `OpenMP <http://www.openmp.org/>`_ to parallelize various
of the computations done in C. galpy can be installed without OpenMP
by specifying the option ``--no-openmp`` when running the installation
commands above::

	   pip install . --install-option="--no-openmp"

or when using pip as follows::

    pip install -U --no-deps --install-option="--no-openmp" git+https://github.com/jobovy/galpy.git#egg=galpy

This might be useful if one is using the
``clang`` compiler, which is the new default on macs with OS X (>=
10.8), but does not support OpenMP. ``clang`` might lead to errors in the
installation of galpy such as::

  ld: library not found for -lgomp

  clang: error: linker command failed with exit code 1 (use -v to see invocation)

If you get these errors, you can use the commands given above to
install without OpenMP, or specify to use ``gcc`` by specifying the
``CC`` and ``LDSHARED`` environment variables to use ``gcc``. Note that recent
versions of ``galpy`` attempt to automatically detect OpenMP support, so using
``--no-openmp`` should not typically be necessary even on Macs.

.. _configfile:

Configuration file
-------------------

Since v1.2, ``galpy`` uses a configuration file to set a small number
of configuration variables. This configuration file is parsed using
`ConfigParser
<https://docs.python.org/2/library/configparser.html>`__/`configparser
<https://docs.python.org/3/library/configparser.html>`__. It is
currently used:

	  * to set a default set of distance and velocity scales (``ro`` and ``vo`` throughout galpy) for conversion between physical and internal galpy unit

    	  * to decide whether to use seaborn plotting with galpy's defaults (which affects *all* plotting after importing ``galpy.util.plot``),

	  * to specify whether output from functions or methods should be given as an `astropy Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__ with units as much as possible or not, and whether or not to use astropy's `coordinate transformations <http://docs.astropy.org/en/stable/coordinates/index.html>`__ (these are typically somewhat slower than galpy's own coordinate transformations, but they are more accurate and more general)

          * to set the level of verbosity of galpy's warning system (the default ``verbose=False`` turns off non-crucial warnings).

          * To set options related to whether or not to check for new versions of galpy (``do-check= False`` turns all such checks off; ``check-non-interactive`` sets whether or not to do the version check in non-interactive (script) sessions; ``check-non-interactive`` sets the cadence of how often to check for version updates in non-interactive sessions [in days; interactive sessions always check]; ``last-non-interactive-check`` is an internal variable to store when the last check occurred)

The current configuration file therefore looks like this::

	  [normalization]
	  ro = 8.
	  vo = 220.

	  [plot]
	  seaborn-bovy-defaults = False

	  [astropy]
	  astropy-units = False
	  astropy-coords = True

	  [warnings]
	  verbose = False

	  [version-check]
	  do-check = True
	  check-non-interactive = True
	  check-non-interactive-every = 1
	  last-non-interactive-check = 2000-01-01

where ``ro`` is the distance scale specified in kpc, ``vo`` the
velocity scale in km/s, and the setting is to *not* return output as a
Quantity. These are the current default settings.

A user-wide configuration file should be located at
``$HOME/.galpyrc``. This user-wide file can be overridden by a
``$PWD/.galpyrc`` file in the current directory. If no configuration
file is found, the code will automatically write the default
configuration to ``$HOME/.galpyrc``. Thus, after installing galpy, you
can simply use some of its simplest functionality (e.g., integrate an
orbit), and after this the default configuration file will be present
at ``$HOME/.galpyrc``. If you want to change any of the settings (for
example, you want Quantity output), you can edit this file. The
default configuration file can also be found :download:`here
<examples/galpyrc>`.


.. raw:: html

    <script type="text/javascript">
      var platform = "linux";
      if (navigator.userAgent.indexOf("Win") !== -1) {
        platform = "windows";
      }
      if (navigator.userAgent.indexOf("Mac") !== -1) {
        platform = "mac";
      }
      $(document).ready(function(){
        let platformSelectorTabsets= document.querySelectorAll('.platform-selector-tabset');
        let all_tab_nodes, input_nodes, tab_label_nodes, correct_label, hash, correct_input;
        for (let i = 0; i < platformSelectorTabsets.length; i++) {
          all_tab_nodes = platformSelectorTabsets[i].children;
          input_nodes = [...all_tab_nodes].filter(
              child => child.nodeName === "INPUT");
          tab_label_nodes = [...all_tab_nodes].filter(
              child => child.nodeName === "LABEL");
          correct_label = tab_label_nodes.filter(
              label => label.textContent.trim().toLowerCase() === platform)[0];
          hash = correct_label.getAttribute('for');
          correct_input = input_nodes.filter(node => node.id === hash)[0];
          correct_input.checked = true;
        }
      });
     </script>
