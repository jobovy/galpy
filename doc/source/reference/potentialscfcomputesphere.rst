.. _scf_compute_coeffs_sphere:

galpy.potential.scf_compute_coeffs_spherical
=============================================
Note: This function computes Acos and Asin as defined in `Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_, except that we multiply Acos by 2 such that the density from :ref:`Galpy's Hernquist Potential <hernquist_potential>` corresponds to :math:`Acos = \delta_{0n}\delta_{0l}\delta_{0m}`.

Futher note that this function is a specification of :ref:`scf_compute_coeffs_axi <scf_compute_coeffs_axi>` where :math:`Acos_{nlm} = 0` \
at :math:`l\neq0`

For a given :math:`\rho(r)` we can compute :math:`Acos` and :math:`Asin` through the following equation

.. math:: Acos_{nlm}=  \frac{16 \pi a^3}{I_{nl}} \int_{\xi=0}^{\infty}  (1 + \xi)^{2} (1 - \xi)^{-4}  \rho(r) \Phi_{nlm}(\xi) d\xi \qquad Asin_{nlm}=None

Where

.. math:: \Phi_{nlm}(\xi, \cos(\theta)) = -\frac{1}{2 a} (1 - \xi) C_{n}^{3/2}(\xi) \delta_{l0} \delta_{m0}


.. math:: I_{n0} = - K_{n0} \frac{1}{4 a} \frac{(n + 2) (n + 1)}{(n + 3/2)} \qquad K_{nl} = \frac{1}{2}n(n + 3) + 1


:math:`C_{n}^{\alpha}` is the Gegenbauer polynomial.

Also note :math:`\xi = \frac{r - a}{r + a}`, and  :math:`n`, :math:`l` and :math:`m` are integers bounded by :math:`0 <= n < N` , :math:`l = m = 0`

.. autofunction:: galpy.potential.scf_compute_coeffs_spherical
