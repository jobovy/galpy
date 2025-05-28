.. _scf_compute_coeffs:

galpy.potential.scf_compute_coeffs
======================================
Note: This function computes Acos and Asin as defined in `Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_, except that we multiply Acos and Asin by 2 such that the density from :ref:`Galpy's Hernquist Potential <hernquist_potential>` corresponds to :math:`Acos = \delta_{0n}\delta_{0l}\delta_{0m}` and :math:`Asin = 0`.

For a given :math:`\rho(R, z, \phi)` we can compute :math:`Acos` and :math:`Asin` through the following equation

.. math:: \begin{bmatrix}   Acos \\ Asin \end{bmatrix}_{nlm} =  \frac{4 a^3}{I_{nl}} \int_{\xi=0}^{\infty}\int_{\cos(\theta)=-1}^{1}\int_{\phi=0}^{2\pi} (1 + \xi)^{2} (1 - \xi)^{-4}  \rho(R, z, \phi) \Phi_{nlm}(\xi, \cos(\theta), \phi) d\phi d\cos(\theta) d\xi

Where

.. math:: \Phi_{nlm}(\xi, \cos(\theta), \phi) = -\frac{\sqrt{2l + 1}}{a2^{2l + 1}} \sqrt{\frac{(l - m)!}{(l + m)!}} (1 + \xi)^l (1 - \xi)^{l + 1} C_{n}^{2l + 3/2}(\xi) P_{lm}(\cos(\theta)) \begin{bmatrix}   \cos(m\phi) \\ \sin(m\phi) \end{bmatrix}


.. math:: I_{nl} = - K_{nl} \frac{4\pi}{a 2^{8l + 6}} \frac{\Gamma(n + 4l + 3)}{n! (n + 2l + 3/2)[\Gamma(2l + 3/2)]^2} \qquad K_{nl} = \frac{1}{2}n(n + 4l + 3) + (l + 1)(2l + 1)


:math:`P_{lm}` is the Associated Legendre Polynomials whereas :math:`C_{n}^{\alpha}` is the Gegenbauer polynomial.

Also note :math:`\xi = \frac{r - a}{r + a}` , and :math:`n`, :math:`l` and :math:`m` are integers bounded by :math:`0 <= n < N` , :math:`0 <= l < L`, and :math:`0 <= m <= l`



.. autofunction:: galpy.potential.scf_compute_coeffs
