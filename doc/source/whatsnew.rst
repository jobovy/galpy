What's new?
===========

This page gives some of the key improvements in each galpy
version. See the ``HISTORY.txt`` file in the galpy source for full
details on what is new and different in each version.

v1.8
+++++

Version 1.8 contains two big new features and a variety of smaller 
improvements described below. In addition to this, version 1.8 is also the 
first version to fully drop Python 2.7 support (and, thus, all Python 2 
support; note that Python 2 was already almost not supported before). Version 
1.8 also represents the start of a new release cycle, in which we will attempt 
to release a new major version 1.x every year around July 1 and have two minor 
version releases at roughly four-month intervals in between (so around 
November 1 and March 1). Major releases will include this overview of what's 
new since the last major version release.

Major new features:

* ``galpy`` now allows for a very general set of fictitious forces that arise 
  when working in a non-inertial reference frame through the new potential class 
  :ref:`NonInertialFrameForce <noninertialframe_potential>`. The main driver for 
  this new addition is to include the effect of the Milky Way's barycenter 
  acceleration due to the effect of the Large Magellanic Cloud on the orbits of 
  stars, satellite galaxies, and star clusters in the Milky Way. How this can be 
  done exactly is explained in the 
  :ref:`orbit-example-barycentric-acceleration-LMC` section. But a much more 
  general set of non-inertial reference frames are supported: any combination of 
  barycenter acceleration and arbitrary rotations. See 
  :ref:`orbintegration-noninertial` for some more info.

* A particle-spray technique for generating mock stellar streams has been added 
  as :ref:`galpy.df.streamspraydf <api_streamspraydf>`. This roughly follows the 
  `Fardal et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract>`__ 
  implementation, with some notable additions (e.g., the ability to generate a 
  stream around the center of an orbiting satellite). The full ``galpy`` 
  implementation is described in 
  `Qian et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2339Q/abstract>`__.

Other user-facing improvements and additions are

* Potential classes, methods, and functions:

  *  Renamed ``phiforce`` --> ``phitorque`` everywhere (including
     ``potential.evaluatephiforces`` and ``potential.evaluateplanarphiforces``), such 
     that the method's name actually reflect what it returns (a torque, not a force). 
     ``phiforce`` will be fully removed in version 1.9 and may later be re-used 
     for the actual phi component of the force, so switch to the new name now.
  
  * Added ``SCFPotential.from_density`` to directly initialize an ``SCFPotential`` 
    based on a density function. Allows for full correct and consistent handling 
    of Quantity inputs and outputs.
  
  * Added ``TimeDependentAmplitudeWrapperPotential`` for adding arbitrary 
    time-dependence to the amplitude of any Potential/Force.

  * Added ``NullPotential``, a Potential with a constant value (useful, e.g.. 
    to adjust the zero point of a potential, or for testing code in the absence 
    of forces).

  * Added Potential methods/functions ``rE`` and ``LcE`` to compute the radius
    and angular momentum of an orbit with energy E. Also added these 
    as Orbit methods for efficient calculation for collections of 
    orbits.

  * Added the ``offset=`` keyword to ``RotateAndTiltWrapperPotential``, which 
    allows a Potential/Force instance to also be offset from (0,0,0) in 
    addition to being rotated or tilted.  

* New and improved ``Orbit`` methods:

  * Added a progress bar when integrating multiple objects in a single 
    orbit instance (requires ``tqdm``).
  
  * Added ``rE`` and ``LcE`` for the efficient computation of the radius
    and angular momentum of an orbit with energy E (this is efficient for 
    many orbits in a single ``Orbit`` instance; see above).

  * Updated existing and added new phase-space positions for MW satellite 
    galaxies from `Pace et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022arXiv220505699P/abstract>`__.

  * Updated existing and added new phase-space positions for MW globular 
    clusters from `Baumgardt et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.5138B/abstract>`__, 
    `Vasiliev & Baumgardt (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.5978V/abstract>`__, and 
    `Baumgardt & Vasiliev (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.5957B/abstract>`__.

  * Allow actions to be computed for Orbit instances with actionAngle
    methods that don't compute frequencies.

* Updated spherical distribution functions:

  * Added necessary derivatives to allow spherical DFs to be constructed using 
    PowerSphericalPotentialwCutoff and PlummerPotential.

Finally, ``galpy`` can now also be compiled to WebAssembly using the 
``emscripten`` compiler, as part of the ``pyodide`` project. This allows for 
``galpy`` use in the browser without installation at near-C speeds. See 
:ref:`install_pyodide` for more info. This, for example, powers the new "Try 
``galpy``" interactive session on this documentation's home page.

v1.7
+++++

Version 1.7 adds many new features, mainly in the ``galpy.potential`` and
``galpy.df`` modules. The biggest new additions are:

* A general framework for spherical distribution functions defined
  using :math:`f(E,L)` models. Specifically, general solutions for (a)
  isotropic :math:`f(E)` models, (b) :math:`f(E,L)` models with
  constant anisotropy :math:`\beta`, and (c) :math:`f(E,L)` models
  with Osipkov-Merritt-type anisotropy are implemented for any
  potential/density pair (not necessarily self-consistent). These
  distribution functions can be evaluated, sampled exactly, and any
  moment of the distribution function can be calculated. Documentation
  of this is currently available at
  :ref:`api_sphericaldfs`. Distribution functions with constant
  anisotropy require the `JAX <https://github.com/google/jax>`__.

* In addition to the general solution, the distribution function of a
  few well-known distribution functions was added, including (a)
  Hernquist distribution functions that are isotropic, have constant
  anisotropy, or have Osipkov-Merritt type anisotropy; (b) an
  isotropic Plummer profile; (c) the isotropic NFW profile (either
  using the approx. from Widrow 2000 or using an improved
  approximation) and the Osipkov-Merritt NFW profile (new approximate
  form); (d) the King model (also added as a potential as
  KingPotential).

Other new additions include:

* New or improved potentials and :ref:`potential wrappers
  <potwrappers>`:

  * :ref:`interpSphericalPotential <interpsphere>`: general
    class to build interpolated versions of spherical potentials.

  * :ref:`AdiabaticContractionWrapperPotential
    <api_potwrap_adiabaticcontract>`: wrapper potential to
    adiabatically contract a spherical dark-matter halo in response to
    the adiabatic growth of a baryonic component.

  * :ref:`TriaxialGaussianPotential <api_pot_triaxgauss>`: potential
    of a Gaussian stratified on triaxial ellipsoids (`Emsellem et
    al. 1994
    <https://ui.adsabs.harvard.edu/abs/1994A%26A...285..723E/abstract>`__).

  * :ref:`PowerTriaxialPotential <api_pot_triaxpower>`: potential of a
    triaxial power-law density (like ``PowerSphericalPotential``, but
    triaxial).

  * :ref:`AnyAxisymmetricRazorThinDiskPotential
    <api_pot_arbitraryrazorthin>`: potential of an arbitrary
    razor-thin axisymmetric disk (not in C, mainly useful for
    rotation-curve modeling).

  * :ref:`AnySphericalPotential <api_pot_arbitraryspherical>`:
    potential of an arbitrary spherical density distribution (not in
    C, mainly useful for rotation-curve modeling).

  * :ref:`RotateAndTiltWrapperPotential <api_potwrap_rotatetilt>`:
    wrapper potential to re-orient a potential arbitrarily in three
    dimensions.

* Other changes to Potential classes, methods, and functions:

  * Functions to compute the SCF density/potential expansion
    coefficients based on an N-body representation of the density
    (:ref:`scf_compute_coeffs_spherical_nbody
    <scf_compute_coeffs_sphere_nbody>`,
    :ref:`scf_compute_coeffs_axi_nbody
    <scf_compute_coeffs_axi_nbody>`, and
    :ref:`scf_compute_coeffs_nbody <scf_compute_coeffs_nbody>`).

  * An :ref:`NFWPotential <api_pot_nfw>` can now be initialized using
    ``rmax/vmax``, the radius and value of the maximum circular
    velocity.

  * Potential functions and methods to compute the zero-velocity
    curve: ``zvc`` and ``zvc_range``. The latter computes the range in
    R over which the zero-velocity curve is defined, the former gives
    the positive z position on the zero-velocity curve for a given
    radius in this range.

  * ``rhalf`` Potential function/method for computing the half-mass
    radius.

  * ``tdyn`` Potential function/method for computing the dynamical time
    using the average density.

  * ``Potential.mass`` now always returns the mass within a spherical
    shell if only one argument is given. Implemented faster versions
    of many mass implementations using Gauss' theorem (including
    :ref:`SCFPotential <scf_potential>` and :ref:`DiskSCFPotential
    <disk_scf_potential>`).

  * Mixed azimuthal,vertical 2nd derivatives for all non-axisymmetric
    potentials in function ``evaluatephizderivs`` and method
    ``phizderiv``. Now all second derivatives in cylindrical coordinates
    are implemented.

  * Function/method ``plotSurfaceDensities/plotSurfaceDensity`` for
    plotting, you'll never guess, the surface density of a potential.

  * Re-implementation of ``DoubleExponentialDiskPotential`` using the
    double-exponential formula for integrating Bessel functions,
    resulting in a simpler, more accurate, and more stable
    implementation. This potential is now accurate to ~machine
    precision.

  * Potentials are now as much as possible numerically stable at ``r=0``
    and ``r=inf``, meaning that they can be evaluated there.

Other additions and changes include:

  * Added the inverse action-angle transformations for the isochrone
    potential (in :ref:`actionAngleIsochroneInverse
    <api_aa_isochroneinv>`) and for the one-dimensional harmonic
    oscillator (in :ref:`actionAngleHarmonicInverse
    <api_aa_harminv>`). Also added the action-angle calculation for
    the harmonic oscilator in :ref:`actionAngleHarmonic
    <api_aa_harm>`. Why yes, I have been playing around with the
    TorusMapper a bit!

  * Renamed ``galpy.util.bovy_coords`` to ``galpy.util.coords``,
    ``galpy.util.bovy_conversion`` to ``galpy.util.conversion``, and
    ``galpy.util.bovy_plot`` to ``galpy.util.plot`` (but old ``from
    galpy.util import bovy_X`` will keep working for now). Also
    renamed some other internal utility modules in the same way
    (``bovy_symplecticode``, ``bovy_quadpack``, and ``bovy_ars``;
    these are not kept backwards-compatible). Trying to make the code
    a bit less egotistical!

  * Support for Python 3.9.

v1.6
+++++

This version mainly consists of changes to the internal functioning of
``galpy``; some of the new outward-facing features are:

* `ChandrasekharDynamicalFrictionForce
  <reference/potentialchandrasekhardynfric.html>`__ is now implemented
  in C, leading to 100x to 1000x speed-ups for orbit integrations
  using dynamical friction compared to the prior pure-Python version.

* New potentials:

  * `HomogeneousSpherePotential   <reference/potentialhomogsphere.html>`__: the potential of a constant density sphere out to some radius R.

  * `DehnenSphericalPotential <reference/potentialdehnen.html>`__: the
    Dehnen Spherical Potential from `Dehnen (1993)
    <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`__.

  * `DehnenCoreSphericalPotential
    <reference/potentialcoredehnen.html>`__: the Dehnen Spherical
    Potential from `(Dehnen 1993)
    <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`__ with alpha=0
    (corresponding to an inner core).

* Some notable internal changes:

  * Fixed a bug in how ``DiskSCFPotential`` instances are passed to C
    for orbit integration that in particular affected the
    ``McMillan17`` Milky-Way potential (any hole in the surface
    density was effectively ignored in the C code in v1.5).

  * The performance of orbit animations is significantly improved.

  * All main galpy C extensions are now compiled into a single
    shared-object library ``libgalpy``.

  * Binary wheels are now automatically built for Windows, Mac, and
    most major Linux distributions upon every push to the ``master`` 
    (now ``main``) branch and these are automatically uploaded to PyPI 
    upon release. See the :ref:`Installation Instructions <installation>`
    for more info. Binary wheels on Windows are also built for every
    push on AppVeyor, see the :ref:`Windows installation instructions
    <install_win>`.

v1.5
+++++

This version will be the last to support Python 2.7 as this version of Python is `reaching end-of-life on January 1 2020 <https://python3statement.org/>`__.

* This version's highlight is a fully re-written implementation of
  ``galpy.orbit.Orbit`` such that it can now contain and manipulate
  multiple objects at once. ``galpy.orbit.Orbit`` can be initialized
  with an arbitrary shape of input objects in a :ref:`variety of ways
  <orbmultinit>`, manipulated in a manner similar to Numpy arrays, and
  all ``Orbit`` methods work efficiently on ``Orbit`` instances
  containing multiple objects. Some methods, such as :ref:`orbit
  integration <orbintegration>` and those for :ref:`fast orbital
  characterization <fastchar>` are parallelized on multi-core
  machines. ``Orbit`` instances can contain and manipulate millions of
  objects simultaneously now.

* Added the ``galpy.potentials.mwpotentials`` module with various
  Milky-Way-like potentials. Currently included are MWPotential2014,
  McMillan17 for the potential from McMillan (2017), models 1 through
  4 from Dehnen & Binney (1998), and the three models from Irrgang et
  al. (2013). See :ref:`this section of the API documentation
  <potential-mw>` for details.

* Added a (JSON) list with the phase-space coordinates of known
  objects (mainly Milky Way globular clusters and dwarf galaxies) for
  easy :ref:`Orbit.from_name initialization <orbfromname>`. For
  ease of use, Orbit.from_name also supports tab completion for known
  objects in this list in IPython/Jupyter.

* Added ``galpy.potential.to_amuse`` to create an `AMUSE
  <http://www.amusecode.org>`__ representation of any galpy potential,
  :ref:`allowing galpy potentials to be used as external gravitational
  fields in AMUSE N-body simulations <amusepot>`.

* New or improved potentials and :ref:`potential wrappers <potwrappers>`:

  * `MovingObjectPotential <reference/potentialmovingobj.html>`__: Re-wrote ``potential.MovingObjectPotential`` to allow general mass distributions for the moving object, implemented now as standard galpy potentials. Also added a C implementation of this potential for fast orbit integration.

  * `IsothermalDiskPotential <reference/potentialisodisk.html>`__: The one-dimensional potential of an isothermal self-gravitating disk (sech^2 profile).

  * `NumericalPotentialDerivativesMixin <reference/potentialnumericalpotentialderivsmixin.html>`__: a Mixin class to add numerically-computed forces and second derivatives to any Potential class, allowing new potentials to be implmented quickly by only implementing the potential itself and obtaining all forces and second derivatives numerically.

  * `DehnenSmoothWrapperPotential <reference/potentialdehnensmoothwrapper.html>`__: Can now decay rather than grow a potential by setting ``decay=True``.

  * Added support to combine Potential instances or lists thereof through the addition operator. E.g., ``pot= pot1+pot2+pot3`` to create the combined potential of the three component potentials (pot1,pot2,pot3). Each of these components can be a combined potential itself. As before, combined potentials are simply lists of potentials, so this is simply an alternative (and perhaps more intuitive) way to create these lists.

  * Added support to adjust the amplitude of a Potential instance through multiplication of the instance by a number or through division by a numer. E.g., ``pot= 2.*pot1`` returns a Potential instance that is the same as pot1, except that the amplitude is twice larger. Similarly, ``pot= pot1/2.`` decreases the amplitude by a factor of two. This is useful, for example, to quickly change the mass of a potential. Only works for Potential instances, not for lists of Potential instances.

* New or improved ``galpy.orbit.Orbit`` functionality and methods:

  * Added support for 1D orbit integration in C.

  * Added support to plot arbitrary combinations of the basic Orbit attributes by giving them as an expresion (e.g., ``orb.plot(d2='vR*R/r+vz*z/r')``); requires the `numexpr <https://github.com/pydata/numexpr>`__ package.

  * Switched default Sun's vertical height zo parameter for Orbit initialization to be the value of 20.8 pc from `Bennett & Bovy (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.482.1417B>`__.

  * Add Python and C implementation of Dormand-Prince 8(5,3) integrator.

Pre-v1.5
+++++

v1.4
----

* Added dynamical friction as the `ChandrasekharDynamicalFrictionForce
  <reference/potentialchandrasekhardynfric.html>`__ class, an
  implementation of dynamical friction based on the classical
  Chandrasekhar formula (with recent tweaks from the literature to
  better represent the results from N-body simulations).

* A general ``EllipsoidalPotential`` superclass for implementing
  potentials with densities that are constant on ellipsoids (functions
  of :math:`m^2 = x^2 + y^2/b^2 + z^2/c^2`). Also implemented in
  C. Implementing new types of ellipsoidal potentials now only
  requires three simple functions to be defined: the density as a
  function of m, its derivative with respect to m, and its integral
  with respect to m^2. Makes implementing any ellipsoidal potential a
  breeze. See examples in the new-potentials section below.

* New or improved potentials and :ref:`potential wrappers <potwrappers>`:

  * `CorotatingRotationWrapperPotential <reference/potentialcorotwrapper.html>`__: wrapper to make a pattern (e.g., a `SpiralArmsPotential <reference/potentialspiralarms.html>`__) wind up over time such that it is always corotating (see `Hunt et al. (2018) <http://arxiv.org/abs/1806.02832>`_ for an example of this).

  * `GaussianAmplitudeWrapperPotential <reference/potentialgaussampwrapper.html>`__: wrapper to modulate the amplitude of a (list of) ``Potential`` (s) with a Gaussian.

  * `PerfectEllipsoidPotential <reference/potentialperfectellipsoid.html>`__: Potential of a perfect triaxial ellipsoid (`de Zeeuw 1985 <http://adsabs.harvard.edu/abs/1985MNRAS.216..273D>`__).

  * `SphericalShellPotential <reference/potentialsphericalshell.html>`__: Potential of a thin, spherical shell.

  * `RingPotential <reference/potentialring.html>`__: Potential of a circular ring.

  * Re-implemented ``TwoPowerTriaxialPotential``, ``TriaxialHernquistPotential``, ``TriaxialJaffePotential``, and ``TriaxialNFWPotential`` using the general ``EllipsoidalPotential`` class.

* New ``Potential`` methods and functions:

  * Use nested lists of ``Potential`` instances wherever lists of ``Potential`` instances can be used. Allows easy adding of components (e.g., a bar) to previously defined potentials (which may be lists themselves): new_pot= [pot,bar_pot].
  * `rtide <reference/potentialrtides.html>`__ and `ttensor <reference/potentialttensors.html>`__: compute the tidal radius of an object and the full tidal tensor.
  * `surfdens <reference/potentialsurfdens.html>`__ method and `evaluateSurfaceDensities <reference/potentialsurfdensities.html>`__ function to evaluate the surface density up to a given z.
  * `r2deriv <reference/potentialsphr2deriv.html>`__ and `evaluater2derivs <reference/potentialsphr2derivs.html>`__: 2nd derivative wrt spherical radius.
  * `evaluatephi2derivs <reference/potentialphi2derivs.html>`__: second derivative wrt phi.
  * `evaluateRphiderivs <reference/potentialrphiderivs.html>`__: mixed (R,phi) derivative.

* New or improved ``galpy.orbit.Orbit`` functionality and methods:

  * `Orbit.from_name <reference/orbitfromname.html>`__ to initialize an ``Orbit`` instance from an object's name. E.g., ``orb= Orbit.from_name('LMC')``.
  * Orbit initialization without arguments is now the orbit of the Sun.
  * Orbits can be initialized with a `SkyCoord <http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html>`__.
  * Default ``solarmotion=`` parameter is now 'schoenrich' for the Solar motion of `Schoenrich et al. (2010) <http://adsabs.harvard.edu/abs/2010MNRAS.403.1829S>`__.
  * `rguiding <reference/orbitrguiding.html>`__: Guiding-center radius.
  * `Lz <reference/orbitlz.html>`__: vertical component of the angular momentum.
  * If astropy version > 3, `Orbit.SkyCoord <reference/orbitskycoord.html>`__ method returns a SkyCoord object that includes the velocity information and the Galactocentric frame used by the Orbit instance.

* ``galpy.df.jeans`` module with tools for Jeans modeling. Currently only contains the functions `sigmar <reference/dfjeanssigmar.html>`__ and `sigmalos <reference/dfjeanssigmalos.html>`__ to calculate the velocity dispersion in the radial or line-of-sight direction using the spherical Jeans equation in a given potential, density profile, and anisotropy profile (anisotropy can be radially varying).

* Support for compilation on Windows with MSVC.

v1.3
----

* A fast and precise method for approximating an orbit's eccentricity,
  peri- and apocenter radii, and maximum height above the midplane
  using the Staeckel approximation (see `Mackereth & Bovy 2018
  <https://arxiv.org/abs/1802.02592>`__). Can determine
  these parameters to better than a few percent accuracy in as little
  as 10 :math:`\mu\mathrm{s}` per object, more than 1,000 times faster
  than through direct orbit integration. See :ref:`this section
  <fastchar>` of the documentation for more info.

* A general method for modifying ``Potential`` classes through
  potential wrappers---simple classes that wrap existing potentials to modify
  their behavior. See :ref:`this section <potwrappers>` of the
  documentation for examples and :ref:`this section <addwrappot>` for
  information on how to easily define new wrappers. Example wrappers
  include `SolidBodyRotationWrapperPotential
  <reference/potentialsolidbodyrotationwrapper.html>`__ to allow *any*
  potential to rotate as a solid body and
  `DehnenSmoothWrapperPotential
  <reference/potentialsolidbodyrotationwrapper.html>`__ to smoothly
  grow *any* potential. See :ref:`this section of the galpy.potential
  API page <potwrapperapi>` for an up-to-date list of wrappers.

* New or improved potentials:

  * `DiskSCFPotential <reference/potentialdiskscf.html>`__: a general Poisson solver well suited for galactic disks
  * Bar potentials `SoftenedNeedleBarPotential <reference/potentialsoftenedneedle.html>`__ and `FerrersPotential <reference/potentialferrers.html>`__ (latter only in Python for now)
  * 3D spiral arms model `SpiralArmsPotential <reference/potentialspiralarms.html>`__
  * Henon & Heiles (1964) potential `HenonHeilesPotential <reference/potentialhenonheiles.html>`__
  * Triaxial version of `LogarithmicHaloPotential <reference/potentialloghalo.html>`__
  * 3D version of `DehnenBarPotential <reference/potentialdehnenbar.html>`__
  * Generalized version of `CosmphiDiskPotential <reference/potentialcosmphidisk.html>`__

* New or improved ``galpy.orbit.Orbit`` methods:

  * Method to display an animation of an integrated orbit in jupyter notebooks: `Orbit.animate <reference/orbitanimate.html>`__. See :ref:`this section <orbanim>` of the documentation.
  * Improved default method for fast calculation of eccentricity, zmax, rperi, rap, actions, frequencies, and angles by switching to the Staeckel approximation with automatically-estimated approximation parameters.
  * Improved plotting functions: plotting of spherical radius and of arbitrary user-supplied functions of time in Orbit.plot, Orbit.plot3d, and Orbit.animate.

* ``actionAngleStaeckel`` upgrades:

  * ``actionAngleStaeckel`` methods now allow for different focal lengths delta for different phase-space points and for the order of the Gauss-Legendre integration to be specified (default: 10, which is good enough when using actionAngleStaeckel to compute approximate actions etc. for an axisymmetric potential). 
  * Added an option to the estimateDeltaStaeckel function to facilitate the return of an estimated delta parameter at every phase space point passed, rather than returning a median of the estimate at each point. 

* `galpy.df.schwarzschilddf <reference/dfschwarzschild.html>`__:the simple Schwarzschild distribution function for a razor-thin disk (useful for teaching).


v1.2
----

* Full support for providing inputs to all initializations, methods,
  and functions as `astropy Quantity
  <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`__
  with `units <http://docs.astropy.org/en/stable/units/>`__ and for
  providing outputs as astropy Quantities.

* ``galpy.potential.TwoPowerTriaxialPotential``, a set of triaxial
  potentials with iso-density contours that are arbitrary, similar,
  coaxial ellipsoids whose 'radial' density is a (different) power-law
  at small and large radii: 1/m^alpha/(1+m)^beta-alpha (the triaxial
  generalization of TwoPowerSphericalPotential, with flattening in the
  density rather than in the potential; includes triaxial Hernquist
  and NFW potentials.

* ``galpy.potential.SCFPotential``, a class that implements general
  density/potential pairs through the basis expansion approach to
  solving the Poisson equation of Hernquist & Ostriker (1992).  Also
  implemented functions to compute the coefficients for a given
  density function. See more explanation :ref:`here
  <scf_potential_docs>`.

* ``galpy.actionAngle.actionAngleTorus``: an experimental interface to
  Binney & McMillan's TorusMapper code for computing positions and
  velocities for given actions and angles. See the installation
  instructions for how to properly install this. See :ref:`this
  section <aatorus>` and the ``galpy.actionAngle`` API page for
  documentation.

* ``galpy.actionAngle.actionAngleIsochroneApprox`` (Bovy 2014) now
  implemented for the general case of a time-independent potential.

* ``galpy.df.streamgapdf``, a module for modeling the effect of a
  dark-matter subhalo on a tidal stream. See `Sanders et al. (2016)
  <http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S>`__. Also
  includes the fast methods for computing the density along the stream
  and the stream track for a perturbed stream from `Bovy et al. (2016)
  <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1606.03470>`__.

* ``Orbit.flip`` can now flip the velocities of an orbit in-place by
  specifying ``inplace=True``. This allows correct velocities to be
  easily obtained for backwards-integrated orbits.

* ``galpy.potential.PseudoIsothermalPotential``, a standard
  pseudo-isothermal-sphere
  potential. ``galpy.potential.KuzminDiskPotential``, a razor-thin
  disk potential.

* Internal transformations between equatorial and Galactic coordinates
  are now performed by default using astropy's `coordinates
  <http://docs.astropy.org/en/stable/coordinates/index.html>`__
  module. Transformation of (ra,dec) to Galactic coordinates for
  general epochs.

v1.1
----

* Full support for Python 3.

* ``galpy.potential.SnapshotRZPotential``, a potential class that can
  be used to get a frozen snapshot of the potential of an N-body
  simulation.

* Various other potentials: ``PlummerPotential``, a standard Plummer
  potential; ``MN3ExponentialDiskPotential``, an approximation to an
  exponential disk using three Miyamoto-Nagai potentials (`Smith et
  al. 2015 <http://adsabs.harvard.edu/abs/2015MNRAS.448.2934S>`__);
  ``KuzminKutuzovStaeckelPotential``, a Staeckel potential that can be
  used to approximate the potential of a disk galaxy (`Batsleer &
  Dejonghe 1994
  <http://adsabs.harvard.edu/abs/1994A%26A...287...43B>`__).

* Support for converting potential parameters to `NEMO
  <http://bima.astro.umd.edu/nemo/>`__ format and units.

* Orbit fitting in custom sky coordinates.
