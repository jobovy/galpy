Dynamical modeling of tidal streams
++++++++++++++++++++++++++++++++++++

galpy contains tools to model the dynamics of tidal streams, making
extensive use of action-angle variables. As an example, we can model
the dynamics of the following tidal stream (that of Bovy 2014; ). This
movie shows the disruption of a cluster on a GD-1-like orbit around
the Milky Way:

.. raw:: html

   <embed src="http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_orbplane_comov.mpg" AUTOPLAY="false" LOOP="false" width="600" height="515" Pluginspage="http://www.apple.com/quicktime/" CONTROLLER=True></embed>

The blue line is the orbit of the progenitor cluster and the black
points are cluster members. The disruption is shown in an approximate
orbital plane and the movie is comoving with the progenitor cluster.

Streams can be represented by simple dynamical models in action-angle
coordinates. In action-angle coordinates, stream members are stripped
from the progenitor cluster onto orbits specified by a set of actions
:math:`(J_R,J_\phi,J_Z)`, which remain constant after the stars have
been stripped. This is shown in the following movie, which shows the
generation of the stream in action space

.. raw:: html

   <embed src="http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_aai_jrjzlz_debris.mpg" AUTOPLAY="false" LOOP="false" width="600" height="515" Pluginspage="http://www.apple.com/quicktime/" CONTROLLER=True></embed>

The color-coding gives the angular momentum :math:`J_\phi` and the
black dot shows the progenitor orbit. These actions were calculated
using ``galpy.actionAngle.actionAngleIsochroneApprox``. The points
move slightly because of small errors in the action calculation (these
are correlated, so the cloud of points moves coherently because of
calculation errors). The same movie that also shows the actions of
stars in the cluster can be found `here
<http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_aai_jrjzlz.mpg>`_. This
shows that the actions of stars in the cluster are not conserved
(because the self-gravity of the cluster is important), but that the
actions of stream members freeze once they are stripped. The angle
difference between stars in a stream and the progenitor increases
linearly with time, which is shown in the following movie:

.. raw:: html

   <embed src="http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_aai_arazap.mpg" AUTOPLAY="false" LOOP="false" width="600" height="515" Pluginspage="http://www.apple.com/quicktime/" CONTROLLER=True></embed>

where the radial and vertical angle difference with respect to the
progenitor (co-moving at :math:`(\theta_R,\theta_\phi,\theta_Z) =
(\pi,\pi,\pi)`) is shown for each snapshot (the color-coding gives
:math:`\theta_\phi`).

One last movie provides further insight in how a stream evolves over
time. The following movie shows the evolution of the stream in the two
dimensional plane of frequency and angle along the stream (that is,
both are projections of the three dimensional frequencies or angles
onto the angle direction along the stream). The points are color-coded
by the time at which they were removed from the progenitor cluster.

.. raw:: html

   <embed src="http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_aai_oparapar.mpg" AUTOPLAY="false" LOOP="false" width="600" height="515" Pluginspage="http://www.apple.com/quicktime/" CONTROLLER=True></embed>

It is clear that disruption happens in bursts (at pericenter passages)
and that the initial frequency distribution at the time of removal
does not change (much) with time. However, stars removed at larger
frequency difference move away from the cluster faster, such that the
end of the stream is primarily made up of stars with large frequency
differences with respect to the progenitor. This leads to a gradient
in the typical orbit in the stream, and the stream is on average *not*
on a single orbit.


Modeling streams in galpy
-------------------------

In galpy we can model streams using the tools in
``galpy.df.streamdf``. We setup a streamdf instance by specifying the
host gravitational potential ``pot=``, an actionAngle method
(typically ``galpy.actionAngle.actionAngleIsochroneApprox``), a
``galpy.orbit.Orbit`` instance with the position of the progenitor, a
parameter related to the velocity dispersion of the progenitor, and
the time since disruption began. We first import all of the necessary
modules

>>> from galpy.df import streamdf
>>> from galpy.orbit import Orbit
>>> from galpy.potential import LogarithmicHaloPotential
>>> from galpy.actionAngle import actionAngleIsochroneApprox
>>> from galpy.util import bovy_conversion #for unit conversions

setup the potential and actionAngle instances

>>> lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
>>> aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)

define a progenitor Orbit instance

>>> obs= Orbit([1.56148083,0.35081535,-1.15481504,0.88719443,-0.47713334,0.12019596])

and instantiate the streamdf model

>>> sigv= 0.365 #km/s
>>> sdf= streamdf(sigv/220.,progenitor=obs,pot=lp,aA=aAI,leading=True,nTrackChunks=11,tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.))

for a leading stream. This runs in about half a minute on a 2011
Macbook Air. 

We can calculate some simple properties of the stream, such as the
ratio of the largest and second-to-largest eigenvalue of the Hessian
:math:`\partial \mathbf{\Omega} / \partial \mathbf{J}`

>>> sdf.freqEigvalRatio(isotropic=True)
34.450028399901434

or the model's ratio of the largest and second-to-largest eigenvalue
of the model frequency variance matrix

>>> sdf.freqEigvalRatio()
29.625538344985291

The fact that this ratio is so large means that an approximately one
dimensional stream will form.

Similarly, we can calculate the angle between the frequency vector of
the progenitor and of the model mean frequency vector

>>> sdf.misalignment()
-0.49526013844831596

which returns this angle in degrees. We can also calculate the angle
between the frequency vector of the progenitor and the principal
eigenvector of :math:`\partial \mathbf{\Omega} / \partial \mathbf{J}`

>>> sdf.misalignment(isotropic=True)
 1.2825116841963993

(the reason these are obtained by specifying ``isotropic=True`` is
that these would be the ratio of the eigenvalues or the angle if we
assumed that the disrupted materials action distribution were
isotropic).

Calculating the average stream location (track)
-----------------------------------------------

We can display the stream track in various coordinate systems as
follows

>>> sdf.plotTrack(d1='r',d2='z',interp=True,color='k',spread=2,overplot=False,lw=2.,scaleToPhysical=True)

which gives

.. image:: images/sdf_track_rz.png

which shows the track in Galactocentric *R* and *Z* coordinates as
well as an estimate of the spread around the track as the dash-dotted
line. We can overplot the points along the track along which the
:math:`(\mathbf{x},\mathbf{v}) \rightarrow
(\mathbf{\Omega},\boldsymbol{\theta})` transformation and the track
position is explicitly calculated, by turning off the interpolation

>>> sdf.plotTrack(d1='r',d2='z',interp=False,color='k',spread=0,overplot=True,ls='none',marker='o',scaleToPhysical=True)

which gives

.. image:: images/sdf_track_rz_points.png

We can also overplot the orbit of the progenitor

>>> sdf.plotProgenitor(d1='r',d2='z',color='r',overplot=True,ls='--',scaleToPhysical=True)

to give

.. image:: images/sdf_track_rz_progenitor.png

We can do the same in other coordinate systems, for example *X* and
*Z* (as in Figure 1 of Bovy 2014)

>>> sdf.plotTrack(d1='x',d2='z',interp=True,color='k',spread=2,overplot=False,lw=2.,scaleToPhysical=True)
>>> sdf.plotTrack(d1='x',d2='z',interp=False,color='k',spread=0,overplot=True,ls='none',marker='o',scaleToPhysical=True)
>>> sdf.plotProgenitor(d1='x',d2='z',color='r',overplot=True,ls='--',scaleToPhysical=True)
>>> xlim(12.,14.5); ylim(-3.5,7.6)

which gives

.. image:: images/sdf_track_xz.png

or we can calculate the track in observable coordinates, e.g., 

>>> sdf.plotTrack(d1='ll',d2='dist',interp=True,color='k',spread=2,overplot=False,lw=2.)
>>> sdf.plotTrack(d1='ll',d2='dist',interp=False,color='k',spread=0,overplot=True,ls='none',marker='o')
>>> sdf.plotProgenitor(d1='ll',d2='dist',color='r',overplot=True,ls='--')
>>> xlim(155.,255.); ylim(7.5,14.8)

which displays

.. image:: images/sdf_track_ldist.png

Coordinate transformations to physical coordinates are done using
parameters set when initializing the ``sdf`` instance. See the help
for ``?streamdf`` for a complete list of initialization parameters.

Mock stream data generation
----------------------------

We can also easily generate mock data from the stream model. This uses
``streamdf.sample``. For example,

>>> RvR= sdf.sample(n=1000)

which returns the sampled points as a set
:math:`(R,v_R,v_T,Z,v_Z,\phi)` in natural galpy coordinates. We can
plot these and compare them to the track location

>>> sdf.plotTrack(d1='r',d2='z',interp=True,color='b',spread=2,overplot=False,lw=2.,scaleToPhysical=True)
>>> plot(RvR[0]*8.,RvR[3]*8.,'k.',ms=2.) #multiply by the physical distance scale
>>> xlim(12.,16.5); ylim(2.,7.6)

which gives

.. image:: images/sdf_mock_rz.png

Similarly, we can generate mock data in observable coordinates

>>> lb= sdf.sample(n=1000,lb=True)

and plot it

>>> sdf.plotTrack(d1='ll',d2='dist',interp=True,color='b',spread=2,overplot=False,lw=2.)
>>> plot(lb[0],lb[2],'k.',ms=2.)
>>> xlim(155.,235.); ylim(7.5,10.8)

which displays

.. image:: images/sdf_mock_lb.png

We can also just generate mock stream data in frequency-angle coordinates

>>> mockaA= sdf.sample(n=1000,returnaAdt=True)

which returns a tuple with three components: an array with shape [3,N]
of frequency vectors :math:`(\Omega_R,\Omega_\phi,\Omega_Z)`, an array
with shape [3,N] of angle vectors
:math:`(\theta_R,\theta_\phi,\theta_Z)` and :math:`t_s`, the stripping
time. We can plot the vertical versus the radial angle

>>> plot(mockaA[0][0],mockaA[0][2],'k.',ms=2.)

.. image:: images/sdf_mock_aa_oroz.png

or we can plot the magnitude of the angle offset as a function of
stripping time

>>> plot(mockaA[2],numpy.sqrt(numpy.sum((mockaA[1]-numpy.tile(sdf._progenitor_angle,(1000,1)).T)**2.,axis=0)),'k.',ms=2.)

.. image:: images/sdf_mock_aa_adt.png

