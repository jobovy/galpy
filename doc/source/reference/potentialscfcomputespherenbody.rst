.. _scf_compute_coeffs_sphere_nbody:

galpy.potential.scf_compute_coeffs_spherical_nbody
==================================================

This function is the equivalent to :ref:`scf_compute_coeffs_sphere` but computing the coefficients based on an N-body representation of the density.

Note: This function computes Acos and Asin as defined in `Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_, except that we multiply Acos by 2 such that the density from :ref:`Galpy's Hernquist Potential <hernquist_potential>` corresponds to :math:`Acos = \delta_{0n}\delta_{0l}\delta_{0m}`.

Further note that this function is a specification of :ref:`scf_compute_coeffs_nbody` where :math:`Acos_{nlm} = 0` at :math:`l\neq0`

.. autofunction:: galpy.potential.scf_compute_coeffs_spherical_nbody
