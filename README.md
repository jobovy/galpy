#galpy

**Galactic Dynamics in python**

[![Build Status](https://travis-ci.org/jobovy/galpy.png?branch=master)](https://travis-ci.org/jobovy/galpy)

##AUTHOR

Jo Bovy - bovy at ias dot edu

If you find this code useful in your research, please let me know and
consider citing this package. Thanks!


##DOCUMENTATION

See [here](http://jobovy.github.com/galpy)

##INSTALLATION

Standard python setup.py build/install

A few simple tests can be run with

```
nosetests -v -w nose/
```

##DEPENDENCIES

This package requires [NumPy](http://numpy.scipy.org/), [Scipy] (http://www.scipy.org/), and [Matplotlib] (http://matplotlib.sourceforge.net/).

##DISK DF CORRECTIONS

The dehnendf and shudf disk distribution functions can be corrected to
follow the desired surface-mass density and radial-velocity-dispersion
profiles more closely (see
[1999AJ....118.1201D](http://adsabs.harvard.edu/abs/1999AJ....118.1201D)). Calculating
these corrections is expensive, and a large set of precalculated
corrections can be found
[here](https://github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz) \[tar.gz
archive\].