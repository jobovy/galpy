Adding a New Potential in C
===========================

For fast orbit integration and action-angle calculations, potentials
can be implemented in C. This page outlines the steps required.

Overview
--------

To add a C implementation of a potential to galpy:

1. Add the C implementation of the potential to
   ``galpy/potential/potential_c_ext/`` following the template of
   existing potentials.

2. Add the potential to the ``parse_leapfrog_internalPotential``
   function in ``galpy/potential/potential_c_ext/integrateFullOrbit.c``
   (and similar files for planar orbits and 1D orbits).

3. Add a ``_c`` attribute to the Python class that indicates that the
   potential has a C implementation.

4. Add a ``_parse_c`` method to the Python class that translates the
   Python parameters to C parameters.

See the `galpy wiki <https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors>`__
for detailed step-by-step instructions on adding C potentials.

Template
--------

A C potential implementation typically needs to provide functions for:

* ``{PotentialName}Potential_evaluate`` - the potential value
* ``{PotentialName}Potential_Rforce`` - the radial force
* ``{PotentialName}Potential_zforce`` - the vertical force
* ``{PotentialName}Potential_phitorque`` - the azimuthal torque (for non-axisymmetric potentials)

These functions take ``(R, z, phi, t, nargs, args)`` as arguments, where
``args`` is a double array containing the potential parameters.

See existing implementations in ``galpy/potential/potential_c_ext/`` for
examples.
