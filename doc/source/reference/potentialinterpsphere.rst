.. _interpsphere:

Interpolated spherical potential
================================

The ``interpSphericalPotential`` class provides a general interface to
generate interpolated instances of spherical potentials or lists of
such potentials. This interpolated potential can be used in any
function where other three-dimensional galpy potentials can be
used. This includes functions that use ``C`` to speed up
calculations.

The ``interpSphericalPotential`` interpolates the radial force of a
spherical potential and determines the potential and its second
derivative from the base radial-force interpolation object. To set up
an ``interpSphericalPotential`` instance, either provide it with a
function that returns the radial force and the grid to interpolate it
on, as for example,

>>> from galpy import potential
>>> ip= potential.interpSphericalPotential(rforce=lambda r: -1./r,
                        rgrid=numpy.geomspace(0.01,20,101),Phi0=0.)

which sets up an ``interpSphericalPotential`` instance that has the
same radial force as the spherical ``LogarithmicHaloPotential``. If
you have a function that gives the enclosed mass within a given
radius, simply pass it divided by :math:`-r^2` to set up a
``interpSphericalPotential`` instance for this enclosed-mass profile.

Alternatively, you can specify a ``galpy`` potential or list of
potentials and (again) the radial interpolation grid, as for example,

>>> lp= LogarithmicHaloPotential(normalize=1.)
>>> ip= potential.interpSphericalPotential(rforce=lp,
                      rgrid=numpy.geomspace(0.01,20,101))

Note that, because the potential is defined through integration of the
(negative) radial force, we need to specify the potential at the
smallest grid point, which is done through the ``Phi0=`` keyword in
the first example. When using a ``galpy`` potential (or list), this
value is automatically determined.

Also note that the density of the potential is assumed to be zero
outside of the final radial grid point. That is, the potential outside
of the final grid point is :math:`-GM/r` where :math:`M` is the mass
within the final grid point. If during an orbit integration, the orbit strays outside of the interpolation grid, a warning is issued.

.. WARNING::
   The density of a ``interpSphericalPotential`` instance is assumed to be zero outside of the largest radial grid point.

.. autoclass:: galpy.potential.interpSphericalPotential
   :members: __init__
