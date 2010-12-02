Action-angle coordinates
=========================

galpy can calculate actions and angles for spherical and axisymmetric
potentials. These are implemented in a separate module
``galpy.actionAngle``, but the preferred method for accessing these is
as methods of the ``Orbit`` class. This Section briefly explains how
to do this.


Action-angle coordinates for spherical potentials
--------------------------------------------------

Actions, angles, and orbital frequencies for spherical potentials can
be calculated by first instantiating an ``Orbit`` object and then
calling the action-angle functions. Since some of the methods assume
that orbits are given in re-normalized coordinates, i.e.,
``vc(ro)=1``, it is prudent to only call these methods with orbits
that are initialized in this way. For example, we initialized
everything with

>>> from galpy.potential import LogarithmicHaloPotential
>>> lp= LogarithmicHaloPotential(normalize=1.)
>>> from galpy.orbit import Orbit
>>> o= Orbit(vxvv=[1.,0.1,1.1,0.,0.,0.])

and then compute the radial action, specifying the potential to use

>>> o.jr(lp)
array([  7.22774128e-02,   5.06312577e-11])

The return value is the value for jr, as well as an estimate for the
integration error.

Once a call to any of the action-angle methods of ``Orbit`` is done,
the potential used to evaluate the action-angle coordinates cannot be
changed (since various parts of the calculation of these coordinates
get cached for re-use). Therefore, we can compute the radial angle
then as follows

>>> o.wr()

and the other relevant methods

>>> o.jp() #J_phi = L
>>> o.Tr() #The radial period
>>> o.Tp() #The azimuthal period
>>> o.TrTp() #pi*Tr/Tp


Action-angle coordinates for axisymmetric potentials
-----------------------------------------------------

Action-angle coordinates in the symmetry plane can also be computed
for all axisymmetric potentials. Starting where we began for the
previous example, we can initialize

>>> o= Orbit(vxvv=[1.,0.1,1.1,0.,0.,0.])

Now we explicitly make this a planar orbit in the symmetry plane

>>> o= o.toPlanar()

and then we can again calculate action-angle coordinates. E.g.,

>>> o.jr(lp)
array([  7.22774128e-02,   5.06312577e-11])

As expected, this agrees with the value in the example above, since
the orbit is entirely in the symmetry plane.


Example: Evidence for a Lindblad resonance in the Solar neighborhood
---------------------------------------------------------------------

We can use galpy to calculate action-angle coordinates for a set of
stars in the Solar neighborhood and look for unexplained features. For
this we download the data from the Geneva-Copenhagen Survey
(`2009A&A...501..941H
<http://adsabs.harvard.edu/abs/2009A&A...501..941H>`_; data available
at `viZier
<http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=V/130/>`_). Since
the velocities in this catalog are given as U,V, and W, we use the
``radec`` and ``UVW`` keywords to initialize the orbits from the raw
data.