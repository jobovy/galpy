Installation
==============

galpy can be installed using the standard python ``setup.py`` installation::

      > python setup.py install

or::

	> python setup.py install --prefix=~/local

for a local installation. A basic installation works with just the
numpy/scipy/matplotlib stack.


Advanced installation
----------------------

Certain advanced features require the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_), with action calculations
requiring version 1.14 or higher. On a Mac you can make sure that the
correct architecture is installed using `Homebrew
<http://mxcl.github.com/homebrew/>`_ as::

		> brew install gsl --universal

You should be able to check your version  using::

   > gsl-config --version

Other advanced features, including calculating the normalization of
certain distribution functions using Gauss-Legendre integration
require numpy version 1.7.0 or higher.
