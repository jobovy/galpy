A closer look at orbit integration
======================================

Orbit initialization
---------------------

Orbits can be initialized in various coordinate frames. The simplest
initialization gives the initial conditions directly in the
Galactocentric cylindrical coordinate frame (or in the rectangular
coordinate frame in one dimension). ``Orbit()`` automatically figures
out the dimensionality of the space from the initial conditions in
this case. In three dimensions initial conditions are given either as
``vxvv=[R,vR,vT,z,vz,phi]`` or one can choose not to specify the
azimuth of the orbit and initialize with
``vxvv=[R,vR,vT,z,vz]``. Since potentials in galpy are easily
initialized to have a circular velocity of one at a radius equal to
one, initial coordinates are best given as a fraction of the radius at
which one specifies the circular velocity, and initial velocities are
best expressed as fractions of this circular velocity. For example,

>>> o= Orbit(vxvv=[1.,0.1,1.1,0.,0.1,0.])

initializes a fully three-dimensional orbit, while

>>> o= Orbit(vxvv=[1.,0.1,1.1,0.,0.1])

initializes an orbit in which the azimuth is not tracked, as might be
useful for axisymmetric potentials.

In two dimensions, we can similarly specify fully two-dimensional
orbits ``vxvv=[R,vR,vT,phi]`` or choose not to track the azimuth and
initialize as ``vxvv=[R,vR,vT]``. 

In one dimension we simply initialize as ``vxvv=[x,vx]``.

For orbit integration and characterization of observed stars or
clusters, initial conditions can also be specified directly as
observed quantities when ``radec=True`` is set. In this case a full
three-dimensional orbit is initialized as
``vxvv=[RA,Dec,distance,pmRA,pmDec,Vlos]`` where RA and Dec are
expressed in degrees, the distance is expressed in kpc, proper motions
are expressed in mas/yr (pmra = pmra' * cos[Dec] ), and the
line-of-sight velocity is given in km/s. These observed coordinates
are translated to the Galactocentric cylindrical coordinate frame by
assuming a Solar motion that can be specified as either
``solarmotion=hogg`` (default; `2005ApJ...629..268H
<http://adsabs.harvard.edu/abs/2005ApJ...629..268H>`_),
``solarmotion=dehnen`` (`1998MNRAS.298..387D
<http://adsabs.harvard.edu/abs/1998MNRAS.298..387D>`_) or
``solarmotion=shoenrich`` (`2010MNRAS.403.1829S
<http://adsabs.harvard.edu/abs/2010MNRAS.403.1829S>`_). A circular
velocity can be specified as ``vo=235`` in km/s and a value for the
distance between the Galactic center and the Sun can be given as
``ro=8.5`` in kpc. While the inputs are given in physical units, the
orbit is initialized assuming a circular velocity of one at the
distance of the Sun.

ADD EXAMPLE INITIALIZATION.

Orbit integration
---------------------

After an orbit is initialized, we can integrate it for a set of times
``ts``, given as a numpy array. For example, in a simple logarithmic
potential we can do the following

>>> from galpy.potential import LogarithmicHaloPotential
>>> lp= LogarithmicHaloPotential(normalize=1.)
>>> o= Orbit(vxvv=[1.,0.1,1.1,0.,0.1])
>>> import numpy
>>> ts= numpy.linspace(0,100,10000)
>>> o.integrate(ts,lp)

to integrate the orbit from ``t=0`` to ``t=100``, saving the orbit at
10000 instances.

Displaying the orbit
---------------------