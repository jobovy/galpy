Action-Angle Coordinates
========================

galpy can calculate actions, frequencies, and angles for orbits in a
large variety of potentials using multiple methods.

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Introduction to Action-Angle Coordinates
      :link: introduction
      :link-type: doc

      Overview of action-angle variables, using the Orbit interface,
      and calculating actions for simple potentials.

   .. grid-item-card:: Staeckel Approximation
      :link: staeckel
      :link-type: doc

      The most accurate general method for computing actions in
      axisymmetric potentials, including grid-based interpolation.

   .. grid-item-card:: Adiabatic Approximation
      :link: adiabatic
      :link-type: doc

      Fast but approximate method for thin-disk orbits, including
      grid-based interpolation and comparison with Staeckel.

   .. grid-item-card:: Orbit Integration-based (IsochroneApprox)
      :link: isochroneapprox
      :link-type: doc

      General method for any static potential using orbit integration
      in an auxiliary isochrone potential.

   .. grid-item-card:: Reverse Transformations (actionAngleTorus)
      :link: torus
      :link-type: doc

      Reverse action-angle transformation: from actions and angles
      to phase-space coordinates using the TorusMapper.
