galpy.potential.scf_compute_coeffs_axi
========================================
Note: This function is a specification of `scf_compute_coeffs <potentialscfcompute.html>`__ where :math:`Acos_{nlm} = 0` \
at :math:`m\neq0` and :math:`Asin_{nlm} = None`

For a given :math:`\rho(R, z)` we can compute :math:`Acos` and :math:`Asin` through the following equation

.. math:: Acos_{nlm}=  \frac{4 \pi a^3}{I_{nl}} \int_{\xi=0}^{\infty} \int_{\cos(\theta)=-1}^{1} (1 - \xi)^{-2}  \rho(R, z) \Phi_{nlm}(\xi, \cos(\theta)) d\cos(\theta) d\xi \qquad Asin_{nlm}=None 

Where

.. math:: \Phi_{nlm}(\xi, \cos(\theta)) = -\frac{\sqrt{2l + 1}}{a2^{2l + 2}} (1 + \xi)^l (1 - \xi)^{l + 1} C_{n}^{2l + 3/2}(\xi) P_{l0}(\cos(\theta))  \delta_{m0} 


.. math:: I_{nl} = - K_{nl} \frac{4\pi}{a 2^{8l + 6}} \frac{\Gamma(n + 4l + 3)}{n! (n + 2l + 3/2)[\Gamma(2l + 3/2)]^2} \qquad K_{nl} = \frac{1}{2}n(n + 4l + 3) + (l + 1)(2l + 1)


:math:`P_{lm}` is the Associated Legendre Polynomials whereas :math:`C_{n}^{\alpha}` is the Gegenbauer polynomial.

Also note :math:`\xi = \frac{r - a}{r + a}` 

 :math:`n`, :math:`l` and :math:`m` are integers bounded by :math:`0 <= n < N` , :math:`0 <= l < L`, and :math:`m = 0`

.. autofunction:: galpy.potential.scf_compute_coeffs_axi

