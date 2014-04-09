Installation
==============

galpy can be installed using pip as::

      > pip install galpy

or to upgrade without upgrading the dependencies::

      > pip install -U --no-deps galpy

Some advanced features require the GNU Scientific Library (GSL; see below). If you want to use these, install the GSL first (or install it later and re-install using the upgrade command above).

The latest updates in galpy can be installed using::
    
    > pip install -U --no-deps git+git://github.com/jobovy/galpy.git#egg=galpy

or::

    > pip install -U --no-deps --install-option="--prefix=~/local" git+git://github.com/jobovy/galpy.git#egg=galpy

for a local installation. The latest updates can also be installed from the source code downloaded from github using the standard python ``setup.py`` installation::

      > python setup.py install

or::

	> python setup.py install --prefix=~/local

for a local installation. A basic installation works with just the
numpy/scipy/matplotlib stack. Some basic tests can be performed by executing::

		       > nosetests -v -w nose/


Advanced installation
----------------------

Certain advanced features require the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_), with action calculations
requiring version 1.14 or higher. On a Mac you can make sure that the
correct architecture is installed using `Homebrew
<http://brew.sh/>`_ as::

		> brew install gsl --universal

You should be able to check your version  using::

   > gsl-config --version

Other advanced features, including calculating the normalization of
certain distribution functions using Gauss-Legendre integration
require numpy version 1.7.0 or higher.

galpy uses `OpenMP <http://www.openmp.org/>`_ to parallelize various
of the computations done in C. galpy can be installed without OpenMP
by specifying the option ``--no-openmp`` when running the ``python
setup.py`` commands above or when using pip as follows::

    > pip install -U --no-deps --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

or::

    > pip install -U --no-deps --install-option="--prefix=~/local" --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

for a local installation. This can be especially useful if one is
using the ``clang`` compiler, which is the new default on macs with OS
X (>= 10.8), but does not support OpenMP. This leads to errors in the
installation of galpy such as::

  ld: library not found for -lgomp

  clang: error: linker command failed with exit code 1 (use -v to see invocation)

If you get these errors, you can use the commands given above to
install without OpenMP, or specify to use ``gcc`` by specifying the
``CC`` and ``LDSHARED`` environment variables to use ``gcc``.