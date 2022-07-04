galpy.potential.evaluatePotentials
======================================

.. WARNING::
   ``galpy`` potentials do *not* necessarily approach zero at infinity. To compute, for example, the escape velocity or whether or not an orbit is unbound, you need to take into account the value of the potential at infinity. E.g., :math:`v_{\mathrm{esc}}(r) = \sqrt{2[\Phi(\infty)-\Phi(r)]}`.

.. autofunction:: galpy.potential.evaluatePotentials
