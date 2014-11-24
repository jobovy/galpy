galpy
======

**Galactic Dynamics in python**

.. image:: https://travis-ci.org/jobovy/galpy.svg?branch=master
   :target: http://travis-ci.org/jobovy/galpy

.. image:: https://img.shields.io/coveralls/jobovy/galpy.svg
  :target: https://coveralls.io/r/jobovy/galpy?branch=master

.. image:: http://img.shields.io/badge/C%20coverage-99%-brightgreen.svg
   :target: http://sns.ias.edu/~bovy/galpy_lcov/

.. image:: https://readthedocs.org/projects/galpy/badge/?version=latest
  :target: http://galpy.readthedocs.org/en/latest/

.. image:: http://img.shields.io/pypi/v/galpy.svg
   :target: https://pypi.python.org/pypi/galpy/ 

.. image:: http://img.shields.io/badge/license-New%20BSD-brightgreen.svg
   :target: https://github.com/jobovy/galpy/blob/master/LICENSE

AUTHOR
-------

Jo Bovy - bovy at ias dot edu

See `AUTHORS.txt
<https://github.com/jobovy/galpy/blob/master/AUTHORS.txt>`__ for a
full list of contributors.

If you find this code useful in your research, please let me know. **If
you use galpy in a publication, please cite Bovy (2015, in
preparation) and link to http://github.com/jobovy/galpy**. Please also
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

CONTRIBUTING TO GALPY
----------------------

Some development notes can be found on the `wiki
<http://github.com/jobovy/galpy/wiki/>`__. This includes a list of
small and larger extensions of galpy that would be useful `here
<http://github.com/jobovy/galpy/wiki/Possible-galpy-extensions>`__ as
well as a longer-term roadmap `here
<http://github.com/jobovy/galpy/wiki/Roadmap>`__. Please let the main
developer know if you need any help contributing!

DISK DF CORRECTIONS
--------------------

The dehnendf and shudf disk distribution functions can be corrected to
follow the desired surface-mass density and radial-velocity-dispersion
profiles more closely (see `1999AJ....118.1201D
<http://adsabs.harvard.edu/abs/1999AJ....118.1201D>`__). Calculating
these corrections is expensive, and a large set of precalculated
corrections can be found `here
<http://github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz>`__
\[tar.gz archive\]. Install these by downloading them and unpacking them into the galpy/df_src/data directory before running the setup.py installation. E.g.,

.. code-block:: none

   curl -O https://cloud.github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz
   tar xvzf galpy-dfcorrections.tar.gz -C ./galpy/df_src/data/
