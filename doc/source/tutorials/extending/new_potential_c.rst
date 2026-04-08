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
* ``{PotentialName}Potential_dens`` - the density (optional, for ``hasC_dens``)

C function signatures
---------------------

Each function has the following signature:

.. code-block:: c

   double MyPotentialPotential_evaluate(double R, double z, double phi,
                                        double t,
                                        int nargs, double *args) {
       // args[0], args[1], ... are the potential parameters
       double param1 = args[0];
       double param2 = args[1];
       // Compute and return the potential value (without amp)
       return ...;
   }

   double MyPotentialPotential_Rforce(double R, double z, double phi,
                                      double t,
                                      int nargs, double *args) {
       // Return -dPhi/dR (without amp)
       return ...;
   }

   double MyPotentialPotential_zforce(double R, double z, double phi,
                                      double t,
                                      int nargs, double *args) {
       // Return -dPhi/dz (without amp)
       return ...;
   }

The ``nargs`` parameter gives the number of elements in ``args``, and
``args`` is a ``double`` array containing the potential parameters passed
from Python.

Adding to parse_leapfrog_internalPotential
------------------------------------------

You need to register the new potential in the orbit integration
infrastructure. Edit the ``parse_leapfrog_internalPotential`` function in
``galpy/potential/potential_c_ext/integrateFullOrbit.c`` (and the
corresponding functions in ``integratePlanarOrbit.c`` and
``integrateLinearOrbit.c`` if applicable). Add an ``else if`` block
following the pattern of existing potentials:

.. code-block:: c

   else if ( type == <NEW_POTENTIAL_ID> ) {
       potentialArgs->potentialEval = &MyPotentialPotential_evaluate;
       potentialArgs->Rforce = &MyPotentialPotential_Rforce;
       potentialArgs->zforce = &MyPotentialPotential_zforce;
       potentialArgs->phitorque = &MyPotentialPotential_phitorque;
       potentialArgs->nargs = 2;  // number of parameters
   }

The ``type`` integer ID must match the value used in the Python
``_parse_c`` method.

Python-side attributes and methods
----------------------------------

On the Python class, you need to:

1. Set ``hasC = True`` as a class attribute to indicate that a C
   implementation is available:

   .. code-block:: python

      class MyPotential(Potential):
          hasC = True
          ...

2. Optionally set ``hasC_dxdv = True`` if you implement the phase-space
   derivative functions (for variational orbit integration), and
   ``hasC_dens = True`` if you implement the C density function.

3. Implement a ``_parse_c`` method that returns the integer potential-type
   identifier and a list of parameters to pass to C:

   .. code-block:: python

      def _parse_c(self, ro=None, vo=None):
          return (self._parse_c_type_id,
                  [self._param1, self._param2])

   The type ID is an integer that identifies this potential in the C
   ``parse_leapfrog_internalPotential`` function. Look at the existing
   potentials in ``galpy/potential/`` for the convention used to assign
   these IDs.

Existing implementations as templates
--------------------------------------

The best way to get started is to look at an existing potential that is
similar to yours. Good starting points include:

* ``LogarithmicHaloPotential`` -- a simple axisymmetric potential
  (``LogarithmicHaloPotential.c`` and ``LogarithmicHaloPotential.py``)
* ``MiyamotoNagaiPotential`` -- another straightforward example
* ``SpiralArmsPotential`` -- an example of a non-axisymmetric potential

All C implementations live in ``galpy/potential/potential_c_ext/``.

See existing implementations in ``galpy/potential/potential_c_ext/`` for
examples.
