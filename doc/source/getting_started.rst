Quick-start guide
==================

The most basic features of galpy are its ability to display rotation
curves and perform orbit integration for arbitrary combinations of
potentials. This section introduce the most basic features of
galpy.potential and galpy.orbit.


Rotation curves
---------------

The following code example shows how to initialize a Miyamoto-Nagai disk potential and plot its rotation curve

>>> from galpy.potential import MiyamotoNagaiPotential
>>> mp= MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=1.)
>>> mp.plotRotcurve(Rrange=[0.01,10.],grid=1001)

The ``normalize=1.`` option normalizes the potential such that its
circular velocity is 1 at R=1.

Similarly we can initialize other potentials and plot the combined
rotation curve

>>> from galpy.potential import NFWPotential, HernquistPotential
>>> mp= MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=.6)
>>> np= NFWPotential(a=4.5,normalize=.35)
>>> hp= HernquistPotential(a=0.6/8,normalize=0.05)
>>> from galpy.potential import plotRotcurve
>>> plotRotcurve([hp,mp,np],Rrange=[0.01,10.],grid=1001,yrange=[0.,1.2])

The resulting rotation curve is approximately flat. To show the
rotation curves of the three components do

>>> mp.plotRotcurve(Rrange=[0.01,10.],grid=1001,overplot=True)
>>> hp.plotRotcurve(Rrange=[0.01,10.],grid=1001,overplot=True)
>>> np.plotRotcurve(Rrange=[0.01,10.],grid=1001,overplot=True)

You'll see the following

.. image:: images/rotcurve.png