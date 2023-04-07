actionAngle (``galpy.actionAngle``)
===================================

(**x**, **v**) --> (**J**, **O**, **a**)
------------------------------------------

General instance routines
+++++++++++++++++++++++++++

Not necessarily supported for all different types of actionAngle
calculations. These have extra arguments for different ``actionAngle``
modules, so check the documentation of the module-specific functions
for more info (e.g., ``?actionAngleIsochrone.__call__``)

.. toctree::
   :maxdepth: 1

   __call__ <aacall.rst>
   actionsFreqs <aaactionsfreqs.rst>
   actionsFreqsAngles <aaactionsfreqsangles.rst>
   EccZmaxRperiRap <aaecczmaxrperirap.rst>
   turn_physical_off <aaturnphysicaloff.rst>
   turn_physical_on <aaturnphysicalon.rst>

Specific actionAngle modules
++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 1

   actionAngleHarmonic <aaharmonic.rst>
   actionAngleIsochrone <aaisochrone.rst>
   actionAngleSpherical <aaspherical.rst>
   actionAngleAdiabatic <aaadiabatic.rst>
   actionAngleAdiabaticGrid <aaadiabaticgrid.rst>
   actionAngleStaeckel <aastaeckel.rst>
   actionAngleStaeckelGrid <aastaeckelgrid.rst>
   actionAngleIsochroneApprox <aaisochroneapprox.rst>
   actionAngleVertical <aavertical.rst>

(**J**, **a**) --> (**x**, **v**, **O**)
------------------------------------------

General instance routines
+++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 1

   __call__ <aatcall.rst>
   Freqs <aatfreqs.rst>
   xvFreqs <aatxvfreqs.rst>

Specific actionAngle modules
++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 1

   actionAngleHarmonicInverse <aaharmonicinverse.rst>
   actionAngleIsochroneInverse <aaisochroneinverse.rst>
   actionAngleTorus <aatorus.rst>
   actionAngleVerticalInverse <aaverticalinverse.rst>

In addition to the methods listed above, ``actionAngleTorus`` also has
the following methods:

.. toctree::
   :maxdepth: 1

   hessianFreqs <aathessianfreqs.rst>
   xvJacobianFreqs <aatxvjacobianfreqs.rst>

.. WARNING:: While the ``actionAngleTorus`` code can compute the Jacobian and Hessian of the (**J**, **a**) --> (**x**, **v**, **O**) transformation, the accuracy of these does not appear to be very good using the current interface to the TorusMapper code, so care should be taken when using these.
