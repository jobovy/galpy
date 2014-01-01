Three-dimensional disk distribution functions
================================================

galpy contains a fully three-dimensional disk distribution:
``galpy.df.quasiisothermaldf``, which is an approximately isothermal
distribution function expressed in terms of action--angle variables
(see `2010MNRAS.401.2318B
<http://adsabs.harvard.edu/abs/2010MNRAS.401.2318B>`_ and
`2011MNRAS.413.1889B
<http://adsabs.harvard.edu/abs/2011MNRAS.413.1889B>`_). Recent
research shows that this distribution function provides a good model
for the DF of mono-abundance sub-populations (MAPs) of the Milky Way
disk (see `2013MNRAS.434..652T
<http://adsabs.harvard.edu/abs/2013MNRAS.434..652T>`_ and
`2013ApJ...779..115B
<http://adsabs.harvard.edu/abs/2013ApJ...779..115B>`_). This
distribution function family requires action-angle coordinates to
evaluate the DF, so ``galpy.df.quasiisothermaldf`` makes heavy use of
the routines in ``galpy.actionAngle`` (in particular those in
``galpy.actionAngleAdiabatic`` and
``galpy.actionAngle.actionAngleStaeckel``).
