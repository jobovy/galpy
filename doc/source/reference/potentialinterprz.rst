.. _interprz:

Interpolated axisymmetric potential
===================================

The ``interpRZPotential`` class provides a general interface to
generate interpolated instances of general three-dimensional,
axisymmetric potentials or lists of such potentials. This interpolated
potential can be used in any function where other three-dimensional
galpy potentials can be used. This includes functions that use ``C``
to speed up calculations, if the ``interpRZPotential`` instance was
set up with ``enable_c=True``. Initialize as

>>> from galpy import potential
>>> ip= potential.interpRZPotential(potential.MWPotential,interpPot=True)

which sets up an interpolation of the potential itself only. The
potential and all different forces and functions (``dens``, ``vcirc``,
``epifreq``, ``verticalfreq``, ``dvcircdR``) are interpolated
separately and one needs to specify that these need to be interpolated
separately (so, for example, one needs to set ``interpRforce=True`` to
interpolate the radial force, or ``interpvcirc=True`` to interpolate
the circular velocity).

When points outside the grid are requested within the python code, the
instance will fall back on the original (non-interpolated)
potential. However, when the potential is used purely in ``C``, like
during orbit integration in ``C`` or during action--angle evaluations
in ``C``, there is no way for the potential to fall back onto the
original potential and nonsense or NaNs will be returned. Therefore,
when using ``interpRZPotential`` in ``C``, one must make sure that the
whole relevant part of the ``(R,z)`` plane is covered. One more time:

.. WARNING::
   When an interpolated potential is used purely in ``C``, like during orbit integration in ``C`` or during action--angle evaluations in ``C``, there is no way for the potential to fall back onto the original potential and nonsense or NaNs will be returned. Therefore, when using ``interpRZPotential`` in ``C``, one must make sure that the whole relevant part of the ``(R,z)`` plane is covered.

.. autoclass:: galpy.potential.interpRZPotential
   :members: __init__
