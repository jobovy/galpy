Orbits
======

galpy supports orbit integration in arbitrary potentials, with flexible
initialization from various coordinate systems and efficient parallel
computation for large numbers of orbits.

.. grid:: 1 2 2 3
   :gutter: 4

   .. grid-item-card:: Orbit Initialization
      :link: initialization
      :link-type: doc

      Initialize orbits in cylindrical coordinates, with physical
      units, from observed coordinates, or from astropy SkyCoord.

   .. grid-item-card:: Multiple Orbits
      :link: multiple_orbits
      :link-type: doc

      Work with multiple orbits at once: array initialization,
      slicing, reshaping, and parallel integration.

   .. grid-item-card:: Orbits of Known Objects
      :link: known_objects
      :link-type: doc

      Initialize orbits from object names using SIMBAD, load
      collections of globular clusters, satellite galaxies,
      or the solar system.

   .. grid-item-card:: Integration and Plotting
      :link: integration_and_plotting
      :link-type: doc

      Integrate orbits, display various projections, access
      orbital quantities, check energy conservation, and
      use non-inertial frames.

   .. grid-item-card:: Fast Orbit Characterization
      :link: fast_characterization
      :link-type: doc

      Quickly compute eccentricity, peri/apocenter, and
      zmax using the Staeckel approximation without orbit
      integration.
