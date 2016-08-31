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
   :maxdepth: 2

   __call__ <aacall.rst>
   actionsFreqs <aaactionsfreqs.rst>
   actionsFreqsAngles <aaactionsfreqsangles.rst>

Specific actionAngle modules
++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   actionAngleIsochrone <aaisochrone.rst>
   actionAngleSpherical <aaspherical.rst>
   actionAngleAdiabatic <aaadiabatic.rst>
   actionAngleAdiabaticGrid <aaadiabaticgrid.rst>
   actionAngleStaeckel <aastaeckel.rst>
   actionAngleStaeckelGrid <aastaeckelgrid.rst>
   actionAngleIsochroneApprox <aaisochroneapprox.rst>

(**J**, **a**) --> (**x**, **v**, **O**)
------------------------------------------

General instance routines
+++++++++++++++++++++++++++

Currently, only the interface to the TorusMapper code supports this
method. Instance methods are

.. toctree::
   :maxdepth: 2

   __call__ <aatcall.rst>
   Freqs <aatfreqs.rst>
   hessianFreqs <aathessianfreqs.rst>
   xvFreqs <aatxvfreqs.rst>
   xvJacobianFreqs <aatxvjacobianfreqs.rst>

Specific actionAngle modules
++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   actionAngleTorus <aatorus.rst>
