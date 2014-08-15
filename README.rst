galpy
======

**Galactic Dynamics in python**

.. image:: https://travis-ci.org/jobovy/galpy.png?branch=master 
   :target: http://travis-ci.org/jobovy/galpy

.. image:: https://coveralls.io/repos/jobovy/galpy/badge.png?branch=master
  :target: https://coveralls.io/r/jobovy/galpy?branch=master

.. image:: https://readthedocs.org/projects/galpy/badge/?version=latest
  :target: http://galpy.readthedocs.org/en/latest/

.. image:: http://img.shields.io/pypi/v/galpy.svg
   :target: https://pypi.python.org/pypi/galpy/ 

.. image:: http://img.shields.io/pypi/l/galpy.svg
   :target: https://github.com/jobovy/galpy/blob/master/LICENSE
`lcov coverage <http://sns.ias.edu/~bovy/galpy_lcov/>`__

AUTHOR
-------

Jo Bovy - bovy at ias dot edu

If you find this code useful in your research, please let me know. If
you use galpy in a publication, please cite Bovy (2014, in
preparation) and link to http://github.com/jobovy/galpy. Please also
send me a reference to the paper or send a pull request including your
paper in the list of galpy papers on `this page
<http://galpy.readthedocs.org/en/latest/>`__ (this page is at
doc/source/index.rst). Thanks!


DOCUMENTATION
--------------

The latest documentation can be found `here <http://galpy.readthedocs.org/en/latest/>`__. An alternative that is kept up-to-date less frequently is `here <http://jobovy.github.com/galpy>`__.

DEPENDENCIES
-------------

This package requires `Numpy <http://numpy.scipy.org/>`__, `Scipy <http://www.scipy.org/>`__, and `Matplotlib <http://matplotlib.sourceforge.net/>`__. Certain advanced features require the GNU Scientific Library (`GSL <http://www.gnu.org/software/gsl/>`__)

ISSUES
-------

If you find *any* bug in the code, please report these using the `Issue Tracker <http://github.com/jobovy/galpy/issues>`__ or by emailing the maintainer of the code.

DISK DF CORRECTIONS
--------------------

The dehnendf and shudf disk distribution functions can be corrected to
follow the desired surface-mass density and radial-velocity-dispersion
profiles more closely (see
`1999AJ....118.1201D <http://adsabs.harvard.edu/abs/1999AJ....118.1201D>`__). Calculating
these corrections is expensive, and a large set of precalculated
corrections can be found
`here <http://github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz>`__ \[tar.gz
archive\].

DEVELOPMENT
-----------

Some development notes can be found on the `wiki <http://github.com/jobovy/galpy/wiki/>`__.
