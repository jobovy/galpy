.. _potential-api:
Potential
=========

3D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 2

   __call__ <potentialcall.rst>
   dens <potentialdens.rst>
   dvcircdR <potentialdvcircdr.rst>
   epifreq <potentialepifreq.rst>
   flattening <potentialflattening.rst>
   lindbladR <potentiallindbladR.rst>
   mass <potentialmass.rst>
   omegac <potentialomegac.rst>
   phiforce <potentialphiforce.rst>
   phi2deriv <potentialphi2deriv.rst>
   plot <potentialplot.rst>
   plotDensity <potentialplotdensity.rst>
   plotEscapecurve <potentialplotescapecurve.rst>
   plotRotcurve <potentialplotrotcurve.rst>
   R2deriv <potentialr2deriv.rst>
   Rzderiv <potentialrzderiv.rst>
   Rforce <potentialrforce.rst>
   rl <potentialrl.rst>
   toPlanar <potentialtoplanar.rst>
   toVertical <potentialtovertical.rst>
   vcirc <potentialvcirc.rst>
   verticalfreq <potentialverticalfreq.rst>
   vesc <potentialvesc.rst>
   vterm <potentialvterm.rst>
   z2deriv <potentialz2deriv.rst>
   zforce <potentialzforce.rst>

In addition to these, the ``NFWPotential`` also has methods to calculate virial quantities

.. toctree::
   :maxdepth: 2

   conc <potentialconc.rst>
   mvir <potentialmvir.rst>
   rvir <potentialrvir.rst>

General 3D potential routines
+++++++++++++++++++++++++++++

Use as ``method(...)``

.. toctree::
   :maxdepth: 2

   dvcircdR <potentialdvcircdrs.rst>
   epifreq <potentialepifreqs.rst>
   evaluateDensities <potentialdensities.rst>
   evaluatephiforces <potentialphiforces.rst>
   evaluatePotentials <potentialevaluate.rst>
   evaluateR2derivs <potentialr2derivs.rst>
   evaluateRzderivs <potentialrzderivs.rst>
   evaluateRforces <potentialrforces.rst>
   evaluatez2derivs <potentialz2derivs.rst>
   evaluatezforces <potentialzforces.rst>
   flattening <potentialflattenings.rst>
   lindbladR <potentiallindbladRs.rst>
   omegac <potentialomegacs.rst>
   plotDensities <potentialplotdensities.rst>
   plotEscapecurve <potentialplotescapecurves.rst>
   plotPotentials <potentialplots.rst>
   plotRotcurve <potentialplotrotcurves.rst>
   rl <potentialrls.rst>
   vcirc <potentialvcircs.rst>
   verticalfreq <potentialverticalfreqs.rst>
   vesc <potentialvescs.rst>
   vterm <potentialvterms.rst>

Specific potentials
++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   potentialburkert.rst
   potentialdoubleexp.rst
   potentialdoublepowerspher.rst
   potentialjaffe.rst
   potentialflattenedpower.rst
   potentialhernquist.rst
   potentialinterprz.rst
   potentialisochrone.rst
   potentialkepler.rst
   potentialloghalo.rst
   potentialmiyamoto.rst
   potentialmovingobj.rst
   potentialnfw.rst
   potentialpowerspher.rst
   potentialpowerspherwcut.rst
   potentialrazorexp.rst

.. _potential-mw:

In addition to these classes, a simple Milky-Way-like potential fit to
data on the Milky Way is included as
``galpy.potential.MWPotential2014`` (see the ``galpy`` paper for
details). This potential is defined as

>>> bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
>>> mp= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
>>> np= NFWPotential(a=16/8.,normalize=.35)
>>> MWPotential2014= [bp,mp,np]

and can thus be used like any list of ``Potentials``. If one wants to
add the supermassive black hole at the Galactic center, this can be
done by

>>> from galpy.potential import KeplerPotential
>>> from galpy.util import bovy_conversion
>>> MWPotential2014.append(KeplerPotential(amp=4*10**6./bovy_conversion.mass_in_msol(220.,8.)))

for a black hole with a mass of :math:`4\times10^6\,M_{\odot}`.

An older version ``galpy.potential.MWPotential`` of a similar
potential that was *not* fit to data on the Milky Way is defined as

>>> mp= MiyamotoNagaiPotential(a=0.5,b=0.0375,normalize=.6)
>>> np= NFWPotential(a=4.5,normalize=.35)
>>> hp= HernquistPotential(a=0.6/8,normalize=0.05)
>>> MWPotential= [mp,np,hp]

``galpy.potential.MWPotential2014`` supersedes
``galpy.potential.MWPotential``.

2D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 2

   __call__ <potential2dcall.rst>
   phiforce <potential2dphiforce.rst>
   Rforce <potential2drforce.rst>

General axisymmetric potential instance routines
++++++++++++++++++++++++++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 2

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
   :maxdepth: 2

   evaluateplanarphiforces <potential2dphiforces.rst>
   evaluateplanarPotentials <potential2devaluate.rst>
   evaluateplanarRforces <potential2drforces.rst>
   evaluateplanarR2derivs <potential2dr2derivs.rst>
   LinShuReductionFactor <potential2dlinshureductionfactor.rst>
   plotEscapecurve <potentialplotescapecurves.rst>
   plotplanarPotentials <potential2dplots.rst>
   plotRotcurve <potentialplotrotcurves.rst>

Specific potentials
++++++++++++++++++++

All of the 3D potentials above can be used as two-dimensional
potentials in the mid-plane. 

.. toctree::
   :maxdepth: 2

   RZToplanarPotential <potential2dRZtoplanar.rst>

In addition, a two-dimensional bar potential and a two spiral potentials are included

.. toctree::
   :maxdepth: 2

   potentialdehnenbar.rst
   potentialcosmphidisk.rst
   potentialellipticaldisk.rst
   potentiallopsideddisk.rst
   potentialsteadylogspiral.rst
   potentialtransientlogspiral.rst



1D potentials
-------------

General instance routines
+++++++++++++++++++++++++

Use as ``Potential-instance.method(...)``

.. toctree::
   :maxdepth: 2

   __call__ <potential1dcall.rst>
   force <potential1dforce.rst>
   plot <potential1dplot.rst>


General 1D potential routines
+++++++++++++++++++++++++++++

Use as ``method(...)``

.. toctree::
   :maxdepth: 2

   evaluatelinearForces <potential1dforces.rst>
   evaluatelinearPotentials <potential1devaluate.rst>
   plotlinearPotentials <potential1dplots.rst>

Specific potentials
++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   KGPotential <potentialkg.rst>

One-dimensional potentials can also be derived from 3D axisymmetric potentials as the vertical potential at a certain Galactocentric radius

.. toctree::
   :maxdepth: 2

   RZToverticalPotential <potential1dRZtolinear.rst>
