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
the time since disruption began.

>>> from galpy.df import streamdf
>>> from galpy.potential import LogarithmicHaloPotential