galpy
======

**Galactic Dynamics in python**

.. image:: https://travis-ci.org/jobovy/galpy.svg?branch=master
   :target: http://travis-ci.org/jobovy/galpy

.. image:: https://ci.appveyor.com/api/projects/status/wmgs1sq3i7tbtap2/branch/master?svg=true
   :target: https://ci.appveyor.com/project/jobovy/galpy

.. image:: https://img.shields.io/coveralls/jobovy/galpy.svg
  :target: https://coveralls.io/r/jobovy/galpy?branch=master

.. image:: http://codecov.io/github/jobovy/galpy/coverage.svg?branch=master
  :target: http://codecov.io/github/jobovy/galpy?branch=master

.. image:: https://readthedocs.org/projects/galpy/badge/?version=latest
  :target: http://galpy.readthedocs.io/en/latest/

.. image:: http://img.shields.io/pypi/v/galpy.svg
   :target: https://pypi.python.org/pypi/galpy/ 

.. image:: https://anaconda.org/conda-forge/galpy/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/galpy

.. image:: http://img.shields.io/badge/license-New%20BSD-brightgreen.svg
   :target: https://github.com/jobovy/galpy/blob/master/LICENSE

.. image:: http://img.shields.io/badge/DOI-10.1088/0067%2D%2D0049/216/2/29-blue.svg
   :target: http://dx.doi.org/10.1088/0067-0049/216/2/29

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/

.. image:: https://slackin-galpy.herokuapp.com/badge.svg
   :target: https://galpy.slack.com/

.. image:: https://img.shields.io/badge/join-slack-E01563.svg?style=flat&logo=slack&logoWidth=10
   :target: https://slackin-galpy.herokuapp.com

AUTHOR
-------

Jo Bovy - bovy at astro dot utoronto dot ca

See `AUTHORS.txt
<https://github.com/jobovy/galpy/blob/master/AUTHORS.txt>`__ for a
full list of contributors.

If you find this code useful in your research, please let me
know. **If you use galpy in a publication, please cite** `Bovy (2015)
<http://adsabs.harvard.edu/abs/2015ApJS..216...29B>`__ **and link to
http://github.com/jobovy/galpy**. See `the acknowledgement documentation section
<http://galpy.readthedocs.io/en/latest/index.html#acknowledging-galpy>`__
for a more detailed guide to citing parts of the code. Please also
send me a reference to the paper or send a pull request including your
paper in the list of galpy papers on `this page
<http://galpy.readthedocs.io/en/latest/>`__ (this page is at
doc/source/index.rst). Thanks!


LOOKING FOR HELP?
-----------------

The latest documentation can be found `here <http://galpy.readthedocs.io/en/latest/>`__. You can also join the `galpy slack community <https://galpy.slack.com/>`__ for any questions related to `galpy`; join `here <https://slackin-galpy.herokuapp.com>`__.

If you find *any* bug in the code, please report these using the `Issue Tracker <http://github.com/jobovy/galpy/issues>`__ or by joining the `galpy slack community <https://galpy.slack.com/>`__.

If you are having issues with the installation of ``galpy``, please first consult the `Installation FAQ <http://galpy.readthedocs.io/en/latest/installation.html#installation-faq>`__.

PYTHON VERSIONS AND DEPENDENCIES
---------------------------------

``galpy`` supports both Python 2 and 3. Specifically, galpy supports
Python 2.7 and Python 3.5 and 3.6. It may also work on earlier Python
3.* versions, but this has not been extensively tested. Travis CI
builds regularly check support for Python 2.7 and 3.6 (and of 3.5
using a limited set of tests).

This package requires `Numpy <http://numpy.scipy.org/>`__, `Scipy
<http://www.scipy.org/>`__, and `Matplotlib
<http://matplotlib.sourceforge.net/>`__. Certain advanced features
require the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`__), with action calculations
requiring version 1.14 or higher. Use of ``SnapshotRZPotential`` and
``InterpSnapshotRZPotential`` requires `pynbody
<https://github.com/pynbody/pynbody>`__. Support for providing inputs
and getting outputs as Quantities with units is provided through
`astropy <http://www.astropy.org/>`__.

CONTRIBUTING TO GALPY
----------------------

If you are interested in contributing to galpy's development, take a look at `this brief guide <https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors>`__ on the wiki. This will hopefully help you get started!

Some further development notes can be found on the `wiki
<http://github.com/jobovy/galpy/wiki/>`__. This includes a list of
small and larger extensions of galpy that would be useful `here
<http://github.com/jobovy/galpy/wiki/Possible-galpy-extensions>`__ as
well as a longer-term roadmap `here
<http://github.com/jobovy/galpy/wiki/Roadmap>`__. Please let the main
developer know if you need any help contributing!

DETAILED BUILD, COVERAGE, AND DOCUMENTATION STATUS
---------------------------------------------------

**master**:

.. image:: https://travis-ci.org/jobovy/galpy.svg?branch=master
   :target: http://travis-ci.org/jobovy/galpy

.. image:: https://img.shields.io/coveralls/jobovy/galpy.svg
  :target: https://coveralls.io/r/jobovy/galpy?branch=master

.. image:: http://codecov.io/github/jobovy/galpy/coverage.svg?branch=master
  :target: http://codecov.io/github/jobovy/galpy?branch=master

.. image:: https://readthedocs.org/projects/galpy/badge/?branch=master?version=latest
  :target: http://galpy.readthedocs.io/en/master/


**development branch** (if it exists):

.. image:: https://travis-ci.org/jobovy/galpy.svg?branch=dev
   :target: http://travis-ci.org/jobovy/galpy/branches

.. image:: https://img.shields.io/coveralls/jobovy/galpy.svg?branch=dev
  :target: https://coveralls.io/r/jobovy/galpy?branch=dev

.. image:: http://codecov.io/github/jobovy/galpy/coverage.svg?branch=dev
  :target: http://codecov.io/github/jobovy/galpy?branch=dev

.. image:: https://readthedocs.org/projects/galpy/badge/?branch=master?version=latest
  :target: http://galpy.readthedocs.io/en/dev/

DISK DF CORRECTIONS
--------------------

The dehnendf and shudf disk distribution functions can be corrected to
follow the desired surface-mass density and radial-velocity-dispersion
profiles more closely (see `1999AJ....118.1201D
<http://adsabs.harvard.edu/abs/1999AJ....118.1201D>`__). Calculating
these corrections is expensive, and a large set of precalculated
corrections can be found `here
<http://github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz>`__
\[tar.gz archive\]. Install these by downloading them and unpacking them into the galpy/df/data directory before running the setup.py installation. E.g.::

   curl -O https://github.s3.amazonaws.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz
   tar xvzf galpy-dfcorrections.tar.gz -C ./galpy/df/data/
