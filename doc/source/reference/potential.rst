.. _potential-api:
Potential (``galpy.potential``)
===============================

3D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 1

   __add__ <potentialadd.rst>
   __mul__ <potentialmul.rst>
   __call__ <potentialcall.rst>
   dens <potentialdens.rst>
   dvcircdR <potentialdvcircdr.rst>
   epifreq <potentialepifreq.rst>
   flattening <potentialflattening.rst>
   LcE <potentiallce.rst>
   lindbladR <potentiallindbladR.rst>
   mass <potentialmass.rst>
   nemo_accname <potentialnemoaccname.rst>
   nemo_accpars <potentialnemoaccpars.rst>
   omegac <potentialomegac.rst>
   phitorque <potentialphitorque.rst>
   phizderiv <potentialphizderiv.rst>
   phi2deriv <potentialphi2deriv.rst>
   plot <potentialplot.rst>
   plotDensity <potentialplotdensity.rst>
   plotEscapecurve <potentialplotescapecurve.rst>
   plotRotcurve <potentialplotrotcurve.rst>
   plotSurfaceDensity <potentialplotsurfacedensity.rst>
   R2deriv <potentialr2deriv.rst>
   r2deriv <potentialsphr2deriv.rst>
   rE <potentialre.rst>
   Rzderiv <potentialrzderiv.rst>
   Rforce <potentialrforce.rst>
   rforce <potentialsphrforce.rst>
   rhalf <potentialrhalf.rst>
   rl <potentialrl.rst>
   Rphideriv <potentialrphideriv.rst>
   rtide <potentialrtide.rst>
   surfdens <potentialsurfdens.rst>
   tdyn <potentialtdyn.rst>
   toPlanar <potentialtoplanar.rst>
   toVertical <potentialtovertical.rst>
   ttensor <potentialttensor.rst>
   turn_physical_off <potentialturnphysicaloff.rst>
   turn_physical_on <potentialturnphysicalon.rst>
   vcirc <potentialvcirc.rst>
   verticalfreq <potentialverticalfreq.rst>
   vesc <potentialvesc.rst>
   vterm <potentialvterm.rst>
   z2deriv <potentialz2deriv.rst>
   zforce <potentialzforce.rst>
   zvc <potentialzvc.rst>
   zvc_range <potentialzvcrange.rst>

In addition to these, the ``NFWPotential`` also has methods to calculate virial quantities

.. toctree::
   :maxdepth: 1

   conc <potentialconc.rst>
   mvir <potentialmvir.rst>
   rmax <potentialrmax.rst>
   rvir <potentialrvir.rst>
   vmax <potentialvmax.rst>

General 3D potential routines
+++++++++++++++++++++++++++++

Use as ``method(...)``

.. toctree::
   :maxdepth: 1

   dvcircdR <potentialdvcircdrs.rst>
   epifreq <potentialepifreqs.rst>
   evaluateDensities <potentialdensities.rst>
   evaluatephitorques <potentialphitorques.rst>
   evaluatePotentials <potentialevaluate.rst>
   evaluatephizderivs <potentialphizderivs.rst>
   evaluatephi2derivs <potentialphi2derivs.rst>
   evaluateRphiderivs <potentialrphiderivs.rst>
   evaluateR2derivs <potentialr2derivs.rst>
   evaluater2derivs <potentialsphr2derivs.rst>
   evaluateRzderivs <potentialrzderivs.rst>
   evaluateRforces <potentialrforces.rst>
   evaluaterforces <potentialsphrforces.rst>
   evaluateSurfaceDensities <potentialsurfdensities.rst>
   evaluatez2derivs <potentialz2derivs.rst>
   evaluatezforces <potentialzforces.rst>
   flatten <potentialflatten.rst>
   flattening <potentialflattenings.rst>
   LcE <potentiallces.rst>
   lindbladR <potentiallindbladRs.rst>
   mass <potentialmasses.rst>
   nemo_accname <potentialnemoaccnames.rst>
   nemo_accpars <potentialnemoaccparss.rst>
   omegac <potentialomegacs.rst>
   plotDensities <potentialplotdensities.rst>
   plotEscapecurve <potentialplotescapecurves.rst>
   plotPotentials <potentialplots.rst>
   plotRotcurve <potentialplotrotcurves.rst>
   plotSurfaceDensities <potentialplotsurfacedensities.rst>
   rE <potentialres.rst>
   rhalf <potentialrhalfs.rst>
   rl <potentialrls.rst>
   rtide <potentialrtides.rst>
   tdyn <potentialtdyns.rst>
   to_amuse <potentialtoamuses.rst>
   ttensor <potentialttensors.rst>
   turn_physical_off <potentialturnphysicaloffs.rst>
   turn_physical_on <potentialturnphysicalons.rst>
   vcirc <potentialvcircs.rst>
   verticalfreq <potentialverticalfreqs.rst>
   vesc <potentialvescs.rst>
   vterm <potentialvterms.rst>
   zvc <potentialzvcs.rst>
   zvc_range <potentialzvcranges.rst>

In addition to these, the following methods are available to compute expansion coefficients for the ``SCFPotential`` class for a given density

.. toctree::
   :maxdepth: 1

   scf_compute_coeffs <potentialscfcompute.rst>
   scf_compute_coeffs_axi <potentialscfcomputeaxi.rst>
   scf_compute_coeffs_axi_nbody <potentialscfcomputeaxinbody.rst>
   scf_compute_coeffs_nbody <potentialscfcomputenbody.rst>
   scf_compute_coeffs_spherical <potentialscfcomputesphere.rst>
   scf_compute_coeffs_spherical_nbody <potentialscfcomputespherenbody.rst>

Specific potentials
+++++++++++++++++++

All of the following potentials can also be modified by the specific ``WrapperPotentials`` listed :ref:`below <potwrapperapi>`.

Spherical potentials
********************

Spherical potentials in ``galpy`` can be implemented in two ways: a)
directly by inheriting from ``Potential`` and implementing the usual
methods (``_evaluate``, ``_Rforce``, etc.) or b) by inheriting from
the general :ref:`SphericalPotential <sphericalpot>` class and
implementing the functions ``_revaluate(self,r,t=0.)``,
``_rforce(self,r,t=0.)``, ``_r2deriv(self,r,t=0.)``, and
``_rdens(self,r,t=0.)`` that evaluate the potential, radial force,
(minus the) radial force derivative, and density as a function of the
(here natural) spherical radius. For adding a C implementation when
using method b), follow similar steps in C (use
``interpSphericalPotential`` as an example to follow). For historical
reasons, most spherical potentials in ``galpy`` are directly
implemented (option a above), but for new spherical potentials it is
typically easier to follow option b).

Additional spherical potentials can be obtained by setting the axis
ratios equal for the triaxial potentials listed in the section on
ellipsoidal triaxial potentials below.

.. toctree::
   :maxdepth: 1

   potentialanyspherical.rst
   potentialburkert.rst
   potentialdoublepowerspher.rst
   potentialcoredehnen.rst
   potentialdehnen.rst
   potentialeinasto.rst
   potentialhernquist.rst
   potentialhomogsphere.rst
   potentialinterpsphere.rst
   potentialisochrone.rst
   potentialjaffe.rst
   potentialkepler.rst
   potentialking.rst
   potentialnfw.rst
   potentialplummer.rst
   potentialpowerspher.rst
   potentialpowerspherwcut.rst
   potentialpseudoiso.rst
   potentialsphericalshell.rst

Axisymmetric potentials
***********************

Additional axisymmetric potentials can be obtained by setting the x/y
axis ratio equal to 1 for the triaxial potentials listed in the
section on ellipsoidal triaxial potentials below.

.. toctree::
   :maxdepth: 1

   potentialanyaxirazorthin.rst
   potentialdoubleexp.rst
   potentialflattenedpower.rst
   potentialinterprz.rst
   potentialinterpsnapshotrzpotential.rst
   potentialkuzmindisk.rst
   potentialkuzminkutuzov.rst
   potentialloghalo.rst
   potentialmiyamoto.rst
   potential3mn.rst
   potentialrazorexp.rst
   potentialring.rst
   potentialsnapshotrzpotential.rst

Ellipsoidal triaxial potentials
*******************************

``galpy`` has very general support for implementing triaxial (or the
oblate and prolate special cases) of ellipsoidal potentials through
the general :ref:`EllipsoidalPotential <ellipsoidal>` class. These
potentials have densities that are uniform on ellipsoids, thus only
functions of :math:`m^2 = x^2 + \frac{y^2}{b^2}+\frac{z^2}{c^2}`. New
potentials of this type can be implemented by inheriting from this
class and implementing the ``_mdens(self,m)``, ``_psi(self,m)``, and
``_mdens_deriv`` functions for the density, its integral with respect
to :math:`m^2`, and its derivative with respect to m,
respectively. For adding a C implementation, follow similar steps (use
``PerfectEllipsoidPotential`` as an example to follow).

.. toctree::
   :maxdepth: 1

   potentialdoublepowertriaxial.rst
   potentialperfectellipsoid.rst
   potentialpowertriax.rst
   potentialtriaxialgaussian.rst
   potentialtriaxialjaffe.rst
   potentialtriaxialhernquist.rst
   potentialtriaxialnfw.rst

Note that the Ferrers potential listed below is a potential of this
type, but it is currently not implemented using the
``EllipsoidalPotential`` class. Further note that these potentials can
all be rotated in 3D using the ``zvec`` and ``pa`` keywords; however,
more general support for the same behavior is available through the
``RotateAndTiltWrapperPotential`` discussed below and the internal
``zvec``/``pa`` keywords will likely be deprecated in a future
version.

Spiral, bar, other triaxial, and miscellaneous potentials
**********************************************************

.. toctree::
   :maxdepth: 1

   potentialdehnenbar.rst
   potentialferrers.rst
   potentialloghalo.rst
   potentialmovingobj.rst
   potentialnull.rst
   potentialsoftenedneedle.rst
   potentialspiralarms.rst

All ``galpy`` potentials can also be made to rotate using the ``SolidBodyRotationWrapperPotential`` listed in the section on wrapper potentials :ref:`below <potwrapperapi>`.

General Poisson solvers for disks and halos
*******************************************

.. toctree::
   :maxdepth: 1

   potentialdiskmultipole.rst
   potentialmultipole.rst
   potentialdiskscf.rst
   potentialscf.rst

Dissipative forces
*******************

.. toctree::
   :maxdepth: 1

   potentialfdmdynfric.rst
   potentialchandrasekhardynfric.rst

Fictitious forces in non-inertial frames
****************************************

.. toctree::
   :maxdepth: 1

   potentialnoninertialframe.rst

Helper classes
**************

.. toctree::
   :maxdepth: 1

   potentialnumericalpotentialderivsmixin.rst

2D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 1

   __add__ <potential2dadd.rst>
   __mul__ <potential2dmul.rst>
   __call__ <potential2dcall.rst>
   phitorque <potential2dphitorque.rst>
   Rforce <potential2drforce.rst>
   turn_physical_off <potential2dturnphysicaloff.rst>
   turn_physical_on <potential2dturnphysicalon.rst>

General axisymmetric potential instance routines
++++++++++++++++++++++++++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 1

   epifreq <potential2depifreq.rst>
   lindbladR <potential2dlindbladR.rst>
   omegac <potential2domegac.rst>
   plot <potential2dplot.rst>
   plotEscapecurve <potential2dplotescapecurve.rst>
   plotRotcurve <potential2dplotrotcurve.rst>
   vcirc <potential2dvcirc.rst>
   vesc <potential2dvesc.rst>


General 2D potential routines
+++++++++++++++++++++++++++++

Use as ``method(...)``

.. toctree::
   :maxdepth: 1

   evaluateplanarphitorques <potential2dphitorques.rst>
   evaluateplanarPotentials <potential2devaluate.rst>
   evaluateplanarRforces <potential2drforces.rst>
   evaluateplanarR2derivs <potential2dr2derivs.rst>
   flatten <potentialflatten.rst>
   LinShuReductionFactor <potential2dlinshureductionfactor.rst>
   plotEscapecurve <potentialplotescapecurves.rst>
   plotplanarPotentials <potential2dplots.rst>
   plotRotcurve <potentialplotrotcurves.rst>
   turn_physical_off <potentialturnphysicaloffs.rst>
   turn_physical_on <potentialturnphysicalons.rst>

Specific potentials
+++++++++++++++++++

All of the 3D potentials above can be used as two-dimensional
potentials in the mid-plane.

.. toctree::
   :maxdepth: 1

   toPlanarPotential (general) <potential2dtoplanar.rst>
   RZToplanarPotential <potential2dRZtoplanar.rst>

In addition, a two-dimensional bar potential, two spiral potentials, the `Henon & Heiles (1964) <http://adsabs.harvard.edu/abs/1964AJ.....69...73H>`__ potential, and some static non-axisymmetric perturbations are included

.. toctree::
   :maxdepth: 1

   potentialdehnenbar.rst
   potentialcosmphidisk.rst
   potentialellipticaldisk.rst
   potentialhenonheiles.rst
   potentiallopsideddisk.rst
   potentialsteadylogspiral.rst
   potentialtransientlogspiral.rst



1D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 1

   __add__ <potential1dadd.rst>
   __mul__ <potential1dmul.rst>
   __call__ <potential1dcall.rst>
   force <potential1dforce.rst>
   plot <potential1dplot.rst>
   turn_physical_off <potential1dturnphysicaloff.rst>
   turn_physical_on <potential1dturnphysicalon.rst>

General 1D potential routines
+++++++++++++++++++++++++++++

Use as ``method(...)``

.. toctree::
   :maxdepth: 1

   evaluatelinearForces <potential1dforces.rst>
   evaluatelinearPotentials <potential1devaluate.rst>
   flatten <potentialflatten.rst>
   plotlinearPotentials <potential1dplots.rst>
   turn_physical_off <potentialturnphysicaloffs.rst>
   turn_physical_on <potentialturnphysicalons.rst>

Specific potentials
+++++++++++++++++++

.. toctree::
   :maxdepth: 1

   IsothermalDiskPotential <potentialisodisk.rst>
   KGPotential <potentialkg.rst>

One-dimensional potentials can also be derived from 3D axisymmetric potentials as the vertical potential at a certain Galactocentric radius

.. toctree::
   :maxdepth: 1

   toVerticalPotential (general) <potential1dtolinear.rst>
   RZToverticalPotential <potential1dRZtolinear.rst>

.. _potwrapperapi:

Potential wrappers
-------------------

Gravitational potentials in ``galpy`` can also be modified using wrappers, for example, to change their amplitude as a function of time. These wrappers can be applied to *any* ``galpy`` potential (although whether they can be used in C depends on whether the wrapper *and* all of the potentials that it wraps are implemented in C). Multiple wrappers can be applied to the same potential.

Specific wrappers
++++++++++++++++++

.. toctree::
   :maxdepth: 1

   potentialadiabaticcontractwrapper.rst
   potentialtimedependentamplitude.rst
   potentialcorotwrapper.rst
   potentialcylindricallyseparablewrapper.rst
   potentialdehnensmoothwrapper.rst
   potentialgaussampwrapper.rst
   potentialkuzminlikewrapper.rst
   potentialoblatestaeckelwrapper.rst
   potentialsolidbodyrotationwrapper.rst
   potentialrotateandtiltwrapper.rst
