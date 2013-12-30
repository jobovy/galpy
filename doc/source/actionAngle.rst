Action-angle coordinates
=========================

.. WARNING::
   The action-angle modules are currently being re-developed, so much of the documentation on this page is severely out-of-date!

galpy can calculate actions and angles for a large variety of
potentials (any time-independent potential in principle). These are
implemented in a separate module ``galpy.actionAngle``, and the
preferred method for accessing them is through the routines in this
module. There is also some support for accessing the actionAngle
routines as methods of the ``Orbit`` class.

Action-angle coordinates can be calculated for the following
potentials/approximations:

* Isochrone potential
* Spherical potentials
* Adiabatic approximation
* Staeckel approximation
* A general orbit-integration-based technique

There are classes corresponding to these different
potentials/approximations and actions, frequencies, and angles can
typically be calculated using these three methods:

* __call__: returns the actions
* actionsFreqs: returns the actions and the frequencies
* actionsFreqsAngles: returns the actions, frequencies, and angles

These are not all implemented for each of the cases above yet.

The adiabatic and Staeckel approximation have also been implemented in
C, for extremely fast action-angle calculations (see below).

Action-angle coordinates for the isochrone potential
-----------------------------------------------------

The isochrone potential is the only potential for which all of the
actions, frequencies, and angles can be calculated analytically. We
can do this in galpy by doing

>>> from galpy.potential import IsochronePotential
>>> from galpy.actionAngle import actionAngleIsochrone
>>> ip= IsochronePotential(b=1.,normalize=1.)
>>> aAI= actionAngleIsochrone(ip=ip)

``aAI`` is now an instance that can be used to calculate action-angle
variables for the specific isochrone potential ``ip``. Calling this
instance returns :math:`(J_R,L_Z,J_Z)`

>>> aAI(1.,0.1,1.1,0.1,0.) #inputs R,vR,vT,z,vz
(array([ 0.00713759]), array([ 1.1]), array([ 0.00553155]))

or for a more eccentric orbit

>>> aAI(1.,0.5,1.3,0.2,0.1)
(array([ 0.13769498]), array([ 1.3]), array([ 0.02574507]))

Note that we can also specify ``phi``, but this is not necessary

>>> aAI(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.13769498]), array([ 1.3]), array([ 0.02574507]))

We can likewise calculate the frequencies as well

>>> aAI.actionsFreqs(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.13769498]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 1.29136096]),
 array([ 0.79093738]),
 array([ 0.79093738]))

The output is :math:`(J_R,L_Z,J_Z,\Omega_R,\Omega_\phi,\Omega_Z)`. For
any spherical potential, :math:`\Omega_\phi =
\mathrm{sgn}(L_Z)\Omega_Z`, such that the last two frequencies are the
same.

We obtain the angles as well by calling

>>> aAI.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.13769498]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 1.29136096]),
 array([ 0.79093738]),
 array([ 0.79093738]),
 array([ 0.57101518]),
 array([ 5.96238847]),
 array([ 1.24999949]))

The output here is
:math:`(J_R,L_Z,J_Z,\Omega_R,\Omega_\phi,\Omega_Z,\theta_R,\theta_\phi,\theta_Z)`.

To check that these are good action-angle variables, we can calculate
them along an orbit

>>> from galpy.orbit import Orbit
>>> o= Orbit([1.,0.5,1.3,0.2,0.1,0.])
>>> ts= numpy.linspace(0.,100.,1001)
>>> o.integrate(ts,ip)
>>> jfa= aAI.actionsFreqsAngles(o.R(ts),o.vR(ts),o.vT(ts),o.z(ts),o.vz(ts),o.phi(ts))

which works because we can provide arrays for the ``R`` etc. inputs.

We can then check that the actions are constant over the orbit

>>> plot(ts,numpy.log10(numpy.fabs((jfa[0]-numpy.mean(jfa[0])))))
>>> plot(ts,numpy.log10(numpy.fabs((jfa[1]-numpy.mean(jfa[1])))))
>>> plot(ts,numpy.log10(numpy.fabs((jfa[2]-numpy.mean(jfa[2])))))

which gives

.. image:: images/ip-actions.png

The actions are all conserved. The angles increase linearly with time

.. image:: images/ip-tangles.png

Action-angle coordinates for spherical potentials
--------------------------------------------------

Action-angle coordinates for any spherical potential can be calculated
using a few orbit integrations. These are implemented in galpy in the
``actionAngleSpherical`` module. For example, we can do

>>> from galpy.potential import LogarithmicHaloPotential
>>> lp= LogarithmicHaloPotential(normalize=1.)
>>> from galpy.actionAngle import actionAngleSpherical
>>> aAS= actionAngleSpherical(pot=lp)

For the same eccentric orbit as above we find

>>> aAS(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.22022112]), array([ 1.3]), array([ 0.02574507]))
>>> aAS.actionsFreqs(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.22022112]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 0.87630459]),
 array([ 0.60872881]),
 array([ 0.60872881]))
>>> aAS.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.22022112]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 0.87630459]),
 array([ 0.60872881]),
 array([ 0.60872881]),
 array([ 0.40443857]),
 array([ 5.85965048]),
 array([ 1.1472615]))

We can again check that the actions are conserved along the orbit and
that the angles increase linearly with time:

>>> o.integrate(ts,lp)
>>> jfa= aAS.actionsFreqsAngles(o.R(ts),o.vR(ts),o.vT(ts),o.z(ts),o.vz(ts),o.phi(ts),fixed_quad=True)

where we use ``fixed_quad=True`` for a faster evaluation of the
required one-dimensional integrals using Gaussian quadrature. We then
plot the action fluctuations

>>> plot(ts,numpy.log10(numpy.fabs((jfa[0]-numpy.mean(jfa[0])))))
>>> plot(ts,numpy.log10(numpy.fabs((jfa[1]-numpy.mean(jfa[1])))))
>>> plot(ts,numpy.log10(numpy.fabs((jfa[2]-numpy.mean(jfa[2])))))

which gives

.. image:: images/lp-actions.png

showing that the actions are all conserved. The angles again increase
linearly with time

.. image:: images/lp-tangles.png


We can check the spherical action-angle calculations against the
analytical calculations for the isochrone potential. Starting again
from the isochrone potential used in the previous section

>>> ip= IsochronePotential(b=1.,normalize=1.)
>>> aAI= actionAngleIsochrone(ip=ip)
>>> aAS= actionAngleSpherical(pot=ip)

we can compare the actions, frequencies, and angles computed using
both

>>> aAI.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.13769498]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 1.29136096]),
 array([ 0.79093738]),
 array([ 0.79093738]),
 array([ 0.57101518]),
 array([ 5.96238847]),
 array([ 1.24999949]))
>>> aAS.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.)
(array([ 0.13769498]),
 array([ 1.3]),
 array([ 0.02574507]),
 array([ 1.29136096]),
 array([ 0.79093738]),
 array([ 0.79093738]),
 array([ 0.57101518]),
 array([ 5.96238838]),
 array([ 1.2499994]))

or more explicitly comparing the two

>>> [r-s for r,s in zip(aAI.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.),aAS.actionsFreqsAngles(1.,0.5,1.3,0.2,0.1,0.))]
[array([  6.66133815e-16]),
 array([ 0.]),
 array([ 0.]),
 array([ -4.53851845e-10]),
 array([  4.74775219e-10]),
 array([  4.74775219e-10]),
 array([ -1.65965242e-10]),
 array([  9.04759645e-08]),
 array([  9.04759649e-08])]

Action-angle coordinates using the adiabatic approximation
-----------------------------------------------------------

Action-angle coordinates using the Staeckel approximation
-----------------------------------------------------------

Action-angle coordinates using an orbit-integration-based approximation
-------------------------------------------------------------------------

Accessing action-angle coordinates for Orbit instances
----------------------------------------------------------


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
data. For each object ``ii``

>>> o= Orbit(vxvv[ii,:],radec=True,uvw=True,vo=220.,ro=8.)
>>> o= o.toPlanar()

We then calculate the actions and angles for each object in a flat
rotation curve potential

>>> lp= LogarithmicHaloPotential(normalize=1.)
>>> myjr[ii]= o.jr(lp)[0]

etc.

Plotting the radial action versus the angular momentum

>>> plot.bovy_plot(myjp,myjr/2./nu.pi,'k,',xlabel=r'$J_{\phi}$',ylabel=r'$J_R/2\pi$',xrange=[0.7,1.3],yrange=[0.,0.05])

shows a feature in the distribution

.. image:: images/actionAngle-jrjp.png

If instead we use a power-law rotation curve with power-law index 1

>>> pp= PowerSphericalPotential(normalize=1.,alpha=-2.)
>>> myjr[ii]= o.jr(pp)[0]

We find that the distribution is stretched, but the feature remains

.. image:: images/actionAngle-jrjp-power.png

Code for this example can be found :download:`here
<examples/sellwood-jrjp.py>`. For more information see
`2010MNRAS.409..145S
<http://adsabs.harvard.edu/abs/2010MNRAS.409..145S>`_.
