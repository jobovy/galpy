Installation
==============

Dependencies
------------

galpy requires the ``numpy``, ``scipy``, and ``matplotlib`` packages;
these must be installed or the code will not be able to be imported.

Optional dependencies are: ``astropy`` for `Quantity
<http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__
support (used throughout galpy when installed), ``astroquery`` for the
``Orbit.from_name`` initialization method (to initialize using a
celestial object's name), ``numexpr`` for plotting arbitrary
expressions of ``Orbit`` quantities, and `pynbody
<https://github.com/pynbody/pynbody>`__ for use of
``SnapshotRZPotential`` and ``InterpSnapshotRZPotential``.

To be able to use the fast C extensions for orbit integration and
action-angle calculations, the GNU Scientific Library (GSL) needs to
be installed (:ref:`see below <gsl_install>`).

With conda
----------

The easiest way to install the latest released version of galpy is using conda::

    conda install galpy -c conda-forge

or::

	conda config --add channels conda-forge
	conda install galpy

Installing with conda will automatically install the required
dependencies (``numpy``, ``scipy``, and ``matplotlib``) and the GSL,
but not the optional dependencies.

With pip
--------

galpy can also be installed using pip. Some advanced features require
the GNU Scientific Library (GSL; :ref:`see below <gsl_install>`). If
you want to use these, install the GSL first (or install it later and
re-install using the upgrade command above). Then do::

      pip install galpy

or to upgrade without upgrading the dependencies::

      pip install -U --no-deps galpy

Installing with pip will automatically install the required
dependencies (``numpy``, ``scipy``, and ``matplotlib``), but not the
optional dependencies or the GSL (which needs to be installed *before*
the pip installation of galpy).

Latest version
--------------

The latest updates in galpy can be installed using::
    
    pip install -U --no-deps git+git://github.com/jobovy/galpy.git#egg=galpy

or::

    pip install -U --no-deps --install-option="--prefix=~/local" git+git://github.com/jobovy/galpy.git#egg=galpy

for a local installation. The latest updates can also be installed from the source code downloaded from github using the standard python ``setup.py`` installation::

      python setup.py install

or::

	python setup.py install --prefix=~/local

for a local installation.

Installing from a branch
------------------------

If you want to use a feature that is currently only available in a branch, do::

   pip install -U --no-deps git+git://github.com/jobovy/galpy.git@dev#egg=galpy

to, for example, install the ``dev`` branch.

Installing from source on Windows
---------------------------------

Versions >1.3 should be able to be compiled on Windows systems using the Microsoft Visual Studio C compiler (>= 2015). For this you need to first install the GNU Scientific Library (GSL), for example using Anaconda (:ref:`see below <gsl_install>`). Similar to on a UNIX system, you need to set paths to the header and library files where the GSL is located. On Windows, using the CDM commandline, this is done as::

    set INCLUDE=%CONDA_PREFIX%\Library\include;%INCLUDE%
    set LIB=%CONDA_PREFIX%\Library\lib;%LIB%
    set LIBPATH=%CONDA_PREFIX%\Library\lib;%LIBPATH%

If you are using the Windows PowerShell (which newer versions of the
Anaconda prompt might set as the default), do::

    $env:INCLUDE="$env:CONDA_PREFIX\Library\include"
    $env:LIB="$env:CONDA_PREFIX\Library\lib"
    $env:LIBPATH="$env:CONDA_PREFIX\Library\lib"

where in this example ``CONDA_PREFIX`` is the path of your current conda environment (the path that ends in ``\ENV_NAME``). If you have installed the GSL somewhere else, adjust these paths (but do not use ``YOUR_PATH\include\gsl`` or ``YOUR_PATH\lib\gsl`` as the paths, simply use ``YOUR_PATH\include`` and ``YOUR_PATH\lib``).

To compile with OpenMP on Windows, you have to install Intel OpenMP via::

    conda install -c anaconda intel-openmp

and then to compile the code::

   python setup.py install

If you encounter any issue related to OpenMP during compilation, you can do::

    python setup.py install --no-openmp

.. _install_tm:

Installing from source with Intel Compiler
-------------------------------------------

Compiling galpy with an Intel Compiler can give significant
performance improvements on 64-bit Intel CPUs. Moreover students can
obtain a free copy of an Intel Compiler at `this link
<https://software.intel.com/en-us/qualify-for-free-software/student>`__.

To compile the galpy C extensions with the Intel Compiler on 64bit
MaxOS/Linux do::

    python setup.py build_ext --inplace --compiler=intelem

and to compile the galpy C extensions with the Intel Compiler on 64bit
Windows do::

    python setup.py build_ext --inplace --compiler=intel64w

Then you can simply install with::

     python setup.py install

or other similar installation commands, or you can build your own
wheels with::

    python setup.py sdist bdist_wheel

Installing the TorusMapper code
--------------------------------

Since v1.2, ``galpy`` contains a basic interface to the TorusMapper
code of `Binney & McMillan (2016)
<http://adsabs.harvard.edu/abs/2016MNRAS.456.1982B>`__. This interface
uses a stripped-down version of the TorusMapper code, that is not
bundled with the galpy code, but kept in a fork of the original
TorusMapper code. Installation of the TorusMapper interface is
therefore only possible when installing from source after downloading
or cloning the galpy code and using the ``python setup.py install``
method above.

To install the TorusMapper code, *before* running the installation of
galpy, navigate to the top-level galpy directory (which contains the
``setup.py`` file) and do::

	     git clone https://github.com/jobovy/Torus.git galpy/actionAngle/actionAngleTorus_c_ext/torus
	     cd galpy/actionAngle/actionAngleTorus_c_ext/torus
	     git checkout galpy
	     cd -

Then proceed to install galpy using the ``python setup.py install``
technique or its variants as usual.

Installation FAQ
-----------------

What is the required ``numpy`` version?
++++++++++++++++++++++++++++++++++++++++

``galpy`` should mostly work for any relatively recent version of
``numpy``, but some advanced features, including calculating the
normalization of certain distribution functions using Gauss-Legendre
integration require ``numpy`` version 1.7.0 or higher.

I get warnings like "galpyWarning: integrateFullOrbit_c extension module not loaded, because galpy_integrate_c.so image was not found"
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This typically means that the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_) was unavailable during galpy's
installation, causing the C extensions not to be compiled. Most of the
galpy code will still run, but slower because it will run in pure
Python. The code requires GSL versions >= 1.14. If you believe that
the correct GSL version is installed for galpy, check that the library
can be found during installation (see :ref:`below <gsl_cflags>`).

I get the warning "galpyWarning: actionAngleTorus_c extension module not loaded, because galpy_actionAngleTorus_c.so image was not found"
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is typically because the TorusMapper code was not compiled,
because it was unavailable during installation. This code is only
necessary if you want to use
``galpy.actionAngle.actionAngleTorus``. See :ref:`above <install_tm>`
for instructions on how to install the TorusMapper code.

.. _gsl_install:

How do I install the GSL?
++++++++++++++++++++++++++

Certain advanced features require the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_), with action calculations
requiring version 1.14 or higher. The easiest way to install this is using its Anaconda build::

	  conda install -c conda-forge gsl

If you do not want to go that route, on a Mac, the next easiest way to install
the GSL is using `Homebrew <http://brew.sh/>`_ as::

		brew install gsl --universal

You should be able to check your version using (on Mac/Linux)::

   gsl-config --version

On Linux distributions with ``apt-get``, the GSL can be installed using::

   apt-get install libgsl0-dev

.. _gsl_cflags:

The ``galpy`` installation fails because of C compilation errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``galpy``'s installation can fail due to compilation errors, which look like::

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
       export LD_LIBRARY_PATH=-L/usr/lib

or::

	setenv CFLAGS -I/usr/include
	setenv LDFLAGS -L/usr/lib
	setenv LD_LIBRARY_PATH -L/usr/lib

depending on your shell type (change the actual path to the include
and lib directories that have the gsl directory). If you already have
``CFLAGS``, ``LDFLAGS``, and ``LD_LIBRARY_PATH`` defined you just have
to add the ``'-I/usr/include'`` and ``'-L/usr/lib'`` to them.

If you are on a Mac or UNIX system (e.g., Linux), you can find the correct ``CFLAGS`` and ``LDFLAGS``/``LD_LIBRARY_path`` entries by doing::

   gsl-config --cflags
   gsl-config --libs

(don't include the ``-lgsl lgslcblas`` portion of the latter output.)

I have defined ``CFLAGS``, ``LDFLAGS``, and ``LD_LIBRARY_PATH``, but the compiler does not seem to include these and still returns with errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This typically happens if you install using ``sudo``, but have defined the ``CFLAGS`` etc. environment variables without using sudo. Try using ``sudo -E`` instead, which propagates your own environment variables to the ``sudo`` user.

I'm having issues with OpenMP
+++++++++++++++++++++++++++++++

galpy uses `OpenMP <http://www.openmp.org/>`_ to parallelize various
of the computations done in C. galpy can be installed without OpenMP
by specifying the option ``--no-openmp`` when running the ``python
setup.py`` commands above::

	   python setup.py install --no-openmp

or when using pip as follows::

    pip install -U --no-deps --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

or::

    pip install -U --no-deps --install-option="--prefix=~/local" --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

for a local installation. This might be useful if one is using the
``clang`` compiler, which is the new default on macs with OS X (>=
10.8), but does not support OpenMP. ``clang`` might lead to errors in the
installation of galpy such as::

  ld: library not found for -lgomp

  clang: error: linker command failed with exit code 1 (use -v to see invocation)

If you get these errors, you can use the commands given above to
install without OpenMP, or specify to use ``gcc`` by specifying the
``CC`` and ``LDSHARED`` environment variables to use ``gcc``. Note
that ``clang`` does not seem to have this issue anymore in more recent
versions, but it still does not support ``OpenMP``.

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

    	  * to decide whether to use seaborn plotting with galpy's defaults (which affects *all* plotting after importing ``galpy.util.bovy_plot``), 

	  * to specify whether output from functions or methods should be given as an `astropy Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__ with units as much as possible or not, and whether or not to use astropy's `coordinate transformations <http://docs.astropy.org/en/stable/coordinates/index.html>`__ (these are typically somewhat slower than galpy's own coordinate transformations, but they are more accurate and more general)

          * to set the level of verbosity of galpy's warning system (the default ``verbose=False`` turns off non-crucial warnings). 

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

