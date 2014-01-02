DF
===

Two-dimensional Disk distribution functions
---------------------------------------------

Distribution function for orbits in the plane of a galactic
disk. 

General instance routines
+++++++++++++++++++++++++


.. toctree::
   :maxdepth: 2

   __call__ <diskdfcall.rst>
   asymmetricdrift <diskdfasymmetricdrift.rst>
   kurtosisvR <diskdfkurtosisvR.rst>
   kurtosisvT <diskdfkurtosisvT.rst>
   meanvR <diskdfmeanvR.rst>
   meanvT <diskdfmeanvT.rst>
   oortA <diskdfoortA.rst>
   oortB <diskdfoortB.rst>
   oortC <diskdfoortC.rst>
   oortK <diskdfoortK.rst>
   sigma2surfacemass <diskdfsigma2surfacemass.rst>
   sigma2 <diskdfsigma2.rst>
   sigmaR2 <diskdfsigmaR2.rst>
   sigmaT2 <diskdfsigmaT2.rst>
   skewvR <diskdfskewvR.rst>
   skewvT <diskdfskewvT.rst>
   surfacemass <diskdfsurfacemass.rst>
   surfacemassLOS <diskdfsurfacemassLOS.rst>
   targetSigma2 <diskdftargetSigma2.rst>
   targetSurfacemass <diskdftargetSurfacemass.rst>
   targetSurfacemassLOS <diskdftargetSurfacemassLOS.rst>
   vmomentsurfacemass <diskdfvmomentsurfacemass.rst>

Sampling routines
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   sample <diskdfsample.rst>
   sampledSurfacemassLOS <diskdfsampledSurfacemassLOS.rst>
   sampleLOS <diskdfsampleLOS.rst>
   sampleVRVT <diskdfsampleVRVT.rst>


Specific distribution functions
+++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   Dehnen DF <dfdehnen.rst>
   Shu DF <dfshu.rst>


Three-dimensional Disk distribution functions
----------------------------------------------

Distribution functions for orbits in galactic disks, including the
vertical motion for stars reaching large heights above the
plane. Currently only the *quasi-isothermal DF*.

General instance routines
+++++++++++++++++++++++++


.. toctree::
   :maxdepth: 2

   __call__ <quasidfcall.rst>
   density <quasidfdensity.rst>
   estimate_hr <quasidfestimatehr.rst>
   estimate_hsr <quasidfestimatehsr.rst>
   estimate_hsz <quasidfestimatehsz.rst>
   estimate_hz <quasidfestimatehz.rst>
   jmomentdensity <quasidfjmomentdensity.rst>
   meanjr <quasidfmeanjr.rst>
   meanjz <quasidfmeanjz.rst>
   meanlz <quasidfmeanlz.rst>
   meanvR <quasidfmeanvr.rst>
   meanvT <quasidfmeanvt.rst>
   meanvz <quasidfmeanvz.rst>
   pvR <quasidfpvr.rst>
   pvRvT <quasidfpvrvt.rst>
   pvRvz <quasidfpvrvz.rst>
   pvT <quasidfpvt.rst>
   pvTvz <quasidfpvtvz.rst>
   pvz <quasidfpvz.rst>
   sampleV <quasidfsamplev.rst>
   sigmaR2 <quasidfsigmar2.rst>
   sigmaRz <quasidfsigmarz.rst>
   sigmaT2 <quasidfsigmat2.rst>
   sigmaz2 <quasidfsigmaz2.rst>
   surfacemass_z <quasidfsurfacemass_z.rst>
   tilt <quasidftilt.rst>
   vmomentdensity <quasidfvmomentdensity.rst>


Specific distribution functions
+++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   quasiisothermal DF <dfquasiisothermal.rst>


..
	Ergodic distribution functions
	------------------------------

	General instance routines
	+++++++++++++++++++++++++


	.. toctree::
	   :maxdepth: 2

	      __call__ <edfcall.rst>
	         sample <edfsample.rst>

		 Specific distribution functions
		 +++++++++++++++++++++++++++++++

		 .. toctree::
		    :maxdepth: 2

		       Isothermal DF <dfisotherm.rst>

