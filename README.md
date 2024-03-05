<p align="center">
    <a href="http://www.galpy.org" target="_blank"><img src="https://galpy.readthedocs.io/en/latest/_static/galpy-logo-small.gif"></a><br/>
    <b>Galactic Dynamics in python</b>
</p>

[galpy](http://www.galpy.org) is a Python package for galactic dynamics. It supports orbit integration in a variety of potentials, evaluating and sampling various distribution functions, and the calculation of action-angle coordinates for all static potentials. `galpy` is an [astropy](http://www.astropy.org/) [affiliated package](http://www.astropy.org/affiliated/) and provides full support for astropy’s [Quantity](http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html) framework for variables with units.

[![image](https://github.com/jobovy/galpy/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/jobovy/galpy/actions/workflows/build.yml) [![image](https://github.com/jobovy/galpy/actions/workflows/build_windows.yml/badge.svg?branch=main)](https://github.com/jobovy/galpy/actions/workflows/build_windows.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jobovy/galpy/main.svg)](https://results.pre-commit.ci/latest/github/jobovy/galpy/main)
[![image](http://codecov.io/gh/jobovy/galpy/coverage.svg?branch=main)](http://app.codecov.io/gh/jobovy/galpy?branch=main) [![image](https://readthedocs.org/projects/galpy/badge/?version=latest)](http://docs.galpy.org/en/latest/)
[![image](http://img.shields.io/pypi/v/galpy.svg)](https://pypi.python.org/pypi/galpy/) [![image](https://img.shields.io/pypi/pyversions/galpy?logo=python&logoColor=white)](https://pypi.python.org/pypi/galpy/) [![image](https://anaconda.org/conda-forge/galpy/badges/version.svg)](https://anaconda.org/conda-forge/galpy) [![image](https://img.shields.io/github/commits-since/jobovy/galpy/latest)](https://github.com/jobovy/galpy/commits/main)
[![image](http://img.shields.io/badge/license-New%20BSD-brightgreen.svg)](https://github.com/jobovy/galpy/blob/main/LICENSE) [![image](http://img.shields.io/badge/DOI-10.1088/0067%2D%2D0049/216/2/29-blue.svg)](http://dx.doi.org/10.1088/0067-0049/216/2/29) [![image](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) [![image](https://img.shields.io/badge/join-slack-E01563.svg?style=flat&logo=slack&logoWidth=10)](https://join.slack.com/t/galpy/shared_invite/zt-p6upr4si-mX7u8MRdtm~3bW7o8NA_Ww)

AUTHOR
======

Jo Bovy - bovy at astro dot utoronto dot ca

See
[AUTHORS.txt](https://github.com/jobovy/galpy/blob/main/AUTHORS.txt)
for a full list of contributors.

If you find this code useful in your research, please let me know. **If
you use galpy in a publication, please cite** [Bovy
(2015)](http://adsabs.harvard.edu/abs/2015ApJS..216...29B) **and link to
http://github.com/jobovy/galpy**. See [the acknowledgement documentation
section](http://docs.galpy.org/en/latest/index.html#acknowledging-galpy)
for a more detailed guide to citing parts of the code. Thanks!

LOOKING FOR HELP?
=================

The latest documentation can be found
[here](http://docs.galpy.org/en/latest/). You can also join the
[galpy slack community](https://galpy.slack.com/) for any questions
related to `galpy`; join
[here](https://join.slack.com/t/galpy/shared_invite/zt-p6upr4si-mX7u8MRdtm~3bW7o8NA_Ww).

If you find *any* bug in the code, please report these using the [Issue
Tracker](http://github.com/jobovy/galpy/issues) or by joining the [galpy
slack community](https://galpy.slack.com/).

If you are having issues with the installation of `galpy`, please first
consult the [Installation
FAQ](http://docs.galpy.org/en/latest/installation.html#installation-faq).

PYTHON VERSIONS AND DEPENDENCIES
================================

`galpy` supports Python 3. Specifically, galpy supports Python 3.8, 3.9, 3.10, 3.11,
and 3.12. GitHub Actions CI builds regularly check support for
Python 3.12 (and of 3.8, 3.9, 3.10, and 3.11 using a more limited, core set of tests)
on Linux and Windows (and 3.12 on Mac OS). Python 2.7 is no longer supported.

This package requires [Numpy](https://numpy.org/),
[Scipy](http://www.scipy.org/), and
[Matplotlib](http://matplotlib.sourceforge.net/). Certain advanced
features require the GNU Scientific Library
([GSL](http://www.gnu.org/software/gsl/)), with action calculations
requiring version 1.14 or higher. Other optional dependencies include:

* Support for providing inputs and getting outputs as Quantities with units is provided through
[`astropy`](http://www.astropy.org/).
* Querying SIMBAD for the coordinates of an object in the `Orbit.from_name` initialization method requires [`astroquery`](https://astroquery.readthedocs.io/en/latest/).
* Displaying a progress bar for certain operations (e.g., orbit integration of multiple objects at once) requires [`tqdm`](https://github.com/tqdm/tqdm).
* Plotting arbitrary functions of Orbit attributes requires [`numexpr`](https://github.com/pydata/numexpr).
* Speeding up the evaluation of certain functions in the C code requires [`numba`](https://numba.pydata.org/).
* Constant-anisotropy DFs in `galpy.df.constantbetadf` require [`JAX`](https://github.com/google/jax).
* Use of `SnapshotRZPotential` and `InterpSnapshotRZPotential` requires [`pynbody`](https://github.com/pynbody/pynbody).

Other parts of the code may require additional packages and you will be alerted by the code if they are
not installed.

CONTRIBUTING TO GALPY
=====================

If you are interested in contributing to galpy\'s development, take a
look at [this brief
guide](https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors)
on the wiki. This will hopefully help you get started!

Some further development notes can be found on the
[wiki](http://github.com/jobovy/galpy/wiki/). This includes a list of
small and larger extensions of galpy that would be useful
[here](http://github.com/jobovy/galpy/wiki/Possible-galpy-extensions) as
well as a longer-term roadmap
[here](http://github.com/jobovy/galpy/wiki/Roadmap). Please let the main
developer know if you need any help contributing!
