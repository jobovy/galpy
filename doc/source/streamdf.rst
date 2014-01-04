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
move slightly because of small errors in the action calculation. The
angle difference between stars in a stream and the progenitor
increases linearly with time, which is shown in the following movie:

.. raw:: html

   <embed src="http://sns.ias.edu/~bovy/streams/gd1-sim/gd1_evol_aai_arazap.mpg" AUTOPLAY="false" LOOP="false" width="600" height="515" Pluginspage="http://www.apple.com/quicktime/" CONTROLLER=True></embed>

where the radial and vertical angle difference with respect to the
progenitor (co-moving at :math:`(\theta_R,\theta_\phi,\theta_Z) =
(\pi,\pi,\pi)` is shown for each snapshot (the color-coding gives
:math:`\theta_\phi`).


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


Calculating the average stream location (track)
-----------------------------------------------

We can display the stream track in various coordinate systems as
follows

>>> sdf.plotTrack(d1='r',d2='z',interp=True,color='k',spread=2,overplot=False,lw=2.,scaleToPhysical=True)

which gives

.. image:: images/sdf_track_rz.png

which shows the track in Galactocentric *R* and *Z* coordinates. We
can overplot the points along the track along which the
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