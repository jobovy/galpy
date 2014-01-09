galpy
======

**Galactic Dynamics in python**

.. image:: https://travis-ci.org/jobovy/galpy.png?branch=master 
   :target: http://travis-ci.org/jobovy/galpy

AUTHOR
-------

Jo Bovy - bovy at ias dot edu

If you find this code useful in your research, please let me know and
consider citing this package. Thanks!


DOCUMENTATION
--------------

The latest documentation can be found `here <http://galpy.readthedocs.org/en/latest/>`__. An alternative that is kept up-to-date less frequently is `here <http://jobovy.github.com/galpy>`__.

DEPENDENCIES
-------------

This package requires `Numpy <http://numpy.scipy.org/>`__, `Scipy <http://www.scipy.org/>`__, and `Matplotlib <http://matplotlib.sourceforge.net/>`__. Certain advanced features require the GNU Scientific Library (`GSL <http://www.gnu.org/software/gsl/>`__)

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