Potentials in galpy
====================

galpy contains a large variety of potentials in ``galpy.potential``
that can be used for orbit integration, the calculation of
action-angle coordinates, as part of steady-state distribution
functions, and to study the properties of gravitational
potentials. This section introduces some of these features.

Potentials and forces
----------------------

Various 3D and 2D potentials are contained in galpy, list in the
:ref:`API page <potential-api>`. Another way to list the latest overview
of potentials included with galpy is to run

>>> import galpy.potential
>>> print [p for p in dir(galpy.potential) if 'Potential' in p]
['CosmphiDiskPotential',
 'DehnenBarPotential',
 'DoubleExponentialDiskPotential',
 'EllipticalDiskPotential',
 'FlattenedPowerPotential',
 'HernquistPotential',
....]

(list cut here for brevity). Section :ref:`Rotation curves
<rotcurves>` explains how to initialize potentials and how to display
the rotation curve of single Potential instances or of combinations of
such instances. Similarly, we can evaluate a Potential instance

>>> from galpy.potential import MiyamotoNagaiPotential
>>> mp= MiyamotoNagaiPotential(a=0.5,b=0.0375,normalize=1.)
>>> mp(1.,0.)
-1.2889062500000001

Most member functions of Potential instances have corresponding
functions in the galpy.potential module that allow them to be
evaluated for lists of multiple Potential
instances. ``galpy.potential.MWPotential`` is such a list of three
Potential instances

>>> from galpy.potential import MWPotential
>>> print MWPotential
[<galpy.potential_src.MiyamotoNagaiPotential.MiyamotoNagaiPotential instance at 0x1078d5c20>, <galpy.potential_src.TwoPowerSphericalPotential.NFWPotential instance at 0x1078d5c68>, <galpy.potential_src.TwoPowerSphericalPotential.HernquistPotential instance at 0x1078d5cb0>]

and we can evaluate the potential by using the ``evaluatePotentials``
function

>>> from galpy.potential import evaluatePotentials
>>> evaluatePotentials(1.,0.,MWPotential)
-4.5525780402192924

We can plot the potential of axisymmetric potentials (or of
non-axisymmetric potentials at phi=0) using the ``plot`` member
function

>>> mp.plot()

which produces the following plot

.. image:: images/mp-potential.png

Similarly, we can plot combinations of Potentials using
``plotPotentials``, e.g., 

>>> plotPotentials(MWPotential)

.. image:: images/MWPotential-potential.png

These functions have arguments that can provide custom ``R`` and ``z``
ranges for the plot, the number of grid points, the number of
contours, and many other parameters determining the appearance of
these figures.

galpy also allows the forces corresponding to a gravitational
potential to be calculated. Again for the Miyamoto-Nagai Potential
instance from above

>>> mp.Rforce(1.,0.)
-1.0

This value of -1.0 is due to the normalization of the potential such
that the circular velocity is 1. at R=1. Similarly, the vertical force
is zero in the mid-plane

>>> mp.zforce(1.,0.)
-0.0

but not further from the mid-plane

>>> mp.zforce(1.,0.125)
-0.53488743705310848

As explained in :ref:`Units in galpy <units>`, these forces are in
standard galpy units, and we can convert them to physical units using
methods in the ``galpy.util.bovy_conversion`` module. For example,
assuming a physical circular velocity of 220 km/s at R=8 kpc

>>> from galpy.util import bovy_conversion
>>> mp.zforce(1.,0.125)*bovy_conversion.force_in_kmsMyr(220.,8.)
-3.3095671288657584 #km/s/Myr
>>> mp.zforce(1.,0.125)*bovy_conversion.force_in_2piGmsolpc2(220.,8.)
-119.72021771473301 #2 \pi G Msol / pc^2

Again, there are functions in ``galpy.potential`` that allow for the
evaluation of the forces for lists of Potential instances, such that

>>> from galpy.potential import evaluateRforces
>>> evaluateRforces(1.,0.,MWPotential)
-1.0
>>> from galpy.potential import evaluatezforces
>>> evaluatezforces(1.,0.125,MWPotential)*bovy_conversion.force_in_2piGmsolpc2(220.,8.)
>>> -82.898379883714099 #2 \pi G Msol / pc^2

Densities
---------

galpy can also calculate the densities corresponding to gravitational
potentials. For many potentials, the densities are explicitly
implemented, but if they are not, the density is calculated using the
Poisson equation (second derivatives of the potential have to be
implemented for this). For example, for the Miyamoto-Nagai potential,
the density is explicitly implemented

>>> mp.dens(1.,0.)
1.1145444383277576

and we can also calculate this using the Poisson equation

>>> mp.dens(1.,0.,forcepoisson=True)
1.1145444383277574

which are the same to machine precision

>>> mp.dens(1.,0.,forcepoisson=True)-mp.dens(1.,0.)
-2.2204460492503131e-16

Similarly, all of the potentials in ``galpy.potential.MWPotential``
have explicitly-implemented densities, so we can do

>>> from galpy.potential import evaluateDensities
>>> evaluateDensities(1.,0.,MWPotential)
0.71812049194200644

In physical coordinates, this becomes

>>> evaluateDensities(1.,0.,MWPotential)*bovy_conversion.dens_in_msolpc3(220.,8.)
0.1262386383150029 #Msol / pc^3

We can also plot densities

>>> from galpy.potential import plotDensities
>>> plotDensities(MWPotential,rmin=0.1,zmax=0.25,zmin=-0.25,nrs=101,nzs=101)

which gives

.. image:: images/MWPotential-density.png

Another example of this is for an exponential disk potential

>>> from galpy.potential import DoubleExponentialDiskPotential
>>> dp= DoubleExponentialDiskPotential(hr=1./4.,hz=1./20.,normalize=1.)

The density computed using the Poisson equation now requires multiple
numerical integrations, so the agreement between the analytical
density and that computed using the Poisson equation is slightly less good, but still better than a percent

>>> (dp.dens(1.,0.,forcepoisson=True)-dp.dens(1.,0.))/dp.dens(1.,0.)
0.0032522956769123019

The density is

>>> plotDensities(dp,rmin=0.1,zmax=0.25,zmin=-0.25,nrs=101,nzs=101)

.. image:: images/dp-density.png

and the potential is

>>> dp.plot(rmin=0.1,zmin=-0.25,zmax=0.25)

.. image:: images/dp-potential.png

Clearly, the potential is much less flattened than the density.

Close-to-circular orbits and orbital frequencies
-------------------------------------------------

We can also compute the properties of close-to-circular orbits. First
of all, we can calculate the circular velocity and its derivative

>>> mp.vcirc(1.)
1.0
>>> mp.dvcircdR(1.)
-0.163777427566978

or, for lists of Potential instances

>>> from galpy.potential import vcirc
>>> vcirc(MWPotential,1.)
1.0
>>> from galpy.potential import dvcircdR
>>> dvcircdR(MWPotential,1.)
0.012084123754590059

We can also calculate the various frequencies for close-to-circular
orbits. For example, the rotational frequency

>>> mp.omegac(0.8)
1.2784598203204887
>>> from galpy.potential import omegac
>>> omegac(MWPotential,0.8)
1.2389547535552212

and the epicycle frequency

>>> mp.epifreq(0.8)
1.7774973530267848
>>> from galpy.potential import epifreq
>>> epifreq(MWPotential,0.8)
1.8144833328444094

as well as the vertical frequency

>>> mp.verticalfreq(1.0)
3.7859388972001828
>>> from galpy.potential import verticalfreq
>>> verticalfreq(MWPotential,1.)
3.0000000000000004


For close-to-circular orbits, we can also compute the radii of the
Lindblad resonances. For example, for a frequency similar to that of
the Milky Way's bar

>>> mp.lindbladR(5./3.,m='corotation') #args are pattern speed and m of pattern
0.6027911166042229 #~ 5kpc
>>> print mp.lindbladR(5./3.,m=2)
None
>>> mp.lindbladR(5./3.,m=-2)
0.9906190683480501

The ``None`` here means that there is no inner Lindblad resonance, the
``m=-2`` resonance is in the Solar neighborhood (see the section on
the :ref:`Hercules stream <hercules>` in this documentation).


Adding potentials to the galpy framework
-----------------------------------------