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

2. Register the potential in the ``parse_leapFuncArgs_Full`` function in
   ``galpy/orbit/orbit_c_ext/integrateFullOrbit.c`` (and
   ``parse_leapFuncArgs`` in ``integratePlanarOrbit.c`` for planar
   orbits, and in ``integrateLinearOrbit.c`` for 1D orbits if
   applicable).

3. Register the potential in the Python-side ``_parse_pot`` function
   in ``galpy/orbit/integrateFullOrbit.py`` (and the corresponding
   function in ``integratePlanarOrbit.py`` for planar orbits).

4. Set ``hasC = True`` as a class attribute on the Python potential
   class.

See the `galpy wiki <https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors>`__
for detailed step-by-step instructions on adding C potentials.

Template
--------

A C potential implementation typically needs to provide functions for:

* ``{PotentialName}PotentialEval`` -- the potential value
* ``{PotentialName}PotentialRforce`` -- the radial force
* ``{PotentialName}Potentialzforce`` -- the vertical force
* ``{PotentialName}Potentialphitorque`` -- the azimuthal torque (for non-axisymmetric potentials)
* ``{PotentialName}PotentialDens`` -- the density (optional, for ``hasC_dens``)

For planar (2D) potentials, the naming convention is:

* ``{PotentialName}PotentialPlanarRforce`` -- the planar radial force
* ``{PotentialName}PotentialPlanarphitorque`` -- the planar azimuthal torque (for non-axisymmetric potentials)

C function signatures
---------------------

Each 3D function has the following signature (taking a ``struct potentialArg``
rather than ``nargs``/``args``):

.. code-block:: c

   double MyPotentialEval(double R, double Z, double phi,
                          double t,
                          struct potentialArg * potentialArgs) {
       double * args = potentialArgs->args;
       // args[0], args[1], ... are the potential parameters
       double amp = args[0];
       double param1 = args[1];
       // Compute and return the potential value
       return ...;
   }

   double MyPotentialRforce(double R, double Z, double phi,
                            double t,
                            struct potentialArg * potentialArgs) {
       double * args = potentialArgs->args;
       // Return -dPhi/dR
       return ...;
   }

   double MyPotentialzforce(double R, double Z, double phi,
                            double t,
                            struct potentialArg * potentialArgs) {
       double * args = potentialArgs->args;
       // Return -dPhi/dz
       return ...;
   }

The ``potentialArgs->args`` pointer is a ``double`` array containing the
potential parameters passed from Python (via ``_parse_pot``). The amplitude
``amp`` is typically the first element.

For planar potentials, the signature omits ``Z``:

.. code-block:: c

   double MyPotentialPlanarRforce(double R, double phi, double t,
                                  struct potentialArg * potentialArgs) {
       double * args = potentialArgs->args;
       // Return -dPhi/dR
       return ...;
   }

Registering in parse_leapFuncArgs_Full (C side)
------------------------------------------------

You need to register the new potential in the orbit integration
infrastructure. Edit the ``parse_leapFuncArgs_Full`` function in
``galpy/orbit/orbit_c_ext/integrateFullOrbit.c`` (and
``parse_leapFuncArgs`` in ``integratePlanarOrbit.c`` and
``integrateLinearOrbit.c`` if applicable). Add a ``case`` block
in the ``switch`` statement following the pattern of existing potentials:

.. code-block:: c

   case <NEW_POTENTIAL_ID>: // MyPotential, N arguments
       potentialArgs->potentialEval = &MyPotentialEval;
       potentialArgs->Rforce = &MyPotentialRforce;
       potentialArgs->zforce = &MyPotentialzforce;
       potentialArgs->phitorque = &ZeroForce; // or &MyPotentialphitorque
       potentialArgs->nargs = 2;  // number of parameters
       potentialArgs->ntfuncs = 0;
       potentialArgs->requiresVelocity = false;
       break;

The integer ``<NEW_POTENTIAL_ID>`` must match the value used in the Python
``_parse_pot`` function.

Registering in _parse_pot (Python side)
---------------------------------------

On the Python side, edit the ``_parse_pot`` function in
``galpy/orbit/integrateFullOrbit.py``. Add an ``elif`` block that maps
your potential class to its C integer type ID and parameter list:

.. code-block:: python

   elif isinstance(p, potential.MyPotential):
       pot_type.append(N)  # must match the C switch case ID
       pot_args.extend([p._amp, p._param1, p._param2])

The parameters in ``pot_args`` are passed as the ``args`` array in C,
so they must appear in the same order that the C code expects.

Python-side attributes
----------------------

On the Python class, set ``hasC = True`` as a class attribute to indicate
that a C implementation is available:

.. code-block:: python

   class MyPotential(Potential):
       hasC = True
       ...

Optionally set ``hasC_dxdv = True`` if you implement the phase-space
derivative functions (for variational orbit integration), and
``hasC_dens = True`` if you implement the C density function.

Existing implementations as templates
--------------------------------------

The best way to get started is to look at an existing potential that is
similar to yours. Good starting points include:

* ``LogarithmicHaloPotential.c`` -- a 3D axisymmetric potential with
  optional non-axisymmetry
* ``LopsidedDiskPotential.c`` -- a 2D non-axisymmetric potential
* ``MiyamotoNagaiPotential.c`` -- a straightforward 3D example
* ``SpiralArmsPotential.c`` -- a non-axisymmetric potential

All C implementations live in ``galpy/potential/potential_c_ext/``.
