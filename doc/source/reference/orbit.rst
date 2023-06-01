.. _orbit-api:

Orbit (``galpy.orbit``)
========================

See :ref:`Orbit initialization <orbinit>` for a detailed explanation
on how to set up Orbit instances.

Initialization
--------------

.. toctree::
   :maxdepth: 1

   Orbit <orbitinit.rst>
   Orbit.from_fit <orbitfromfit.rst>
   Orbit.from_name <orbitfromname.rst>

Plotting
--------

.. toctree::
   :maxdepth: 1

   animate <orbitanimate.rst>
   animate3d <orbitanimate3d.rst>
   plot <orbitplot.rst>
   plot3d <orbitplot3d.rst>
   plotBruteSOS <orbitplotbrutesos.rst>
   plotSOS <orbitplotsos.rst>

In addition to these methods, any calculable attribute listed below
can be plotted versus other attributes using ``plotATTR``, where
``ATTR`` is an attribute like ``R``, ``ll``, etc. In this case, the y
axis will have ``ATTR`` and the overridable x axis default is
time. For example, ``o.plotR()`` will plot the orbit's R vs time.


Attributes
----------

* .. autoattribute:: galpy.orbit.Orbit.name
     :annotation: Name(s) of objects initialized using Orbit.from_name
* .. autoattribute:: galpy.orbit.Orbit.shape
* .. autoattribute:: galpy.orbit.Orbit.size

Methods
-------

.. toctree::
   :maxdepth: 1

   __call__ <orbitcall.rst>
   __getitem__ <orbitgetitem.rst>
   bb <orbitbb.rst>
   bruteSOS <orbitbrutesos.rst>
   dec <orbitdec.rst>
   dim <orbitdim.rst>
   dist <orbitdist.rst>
   E <orbitE.rst>
   e <orbitecc.rst>
   ER <orbitER.rst>
   Ez <orbitEz.rst>
   flip <orbitflip.rst>
   getOrbit <orbitgetorbit.rst>
   getOrbit_dxdv <orbitgetorbitdxdv.rst>
   helioX <orbitheliox.rst>
   helioY <orbithelioy.rst>
   helioZ <orbithelioz.rst>
   integrate <orbitint.rst>
   integrate_dxdv <orbitintdxdv.rst>
   integrate_SOS <orbitintsos.rst>
   Jacobi <orbitJacobi.rst>
   jp <orbitjp.rst>
   jr <orbitjr.rst>
   jz <orbitjz.rst>
   ll <orbitll.rst>
   L <orbitl.rst>
   LcE <orbitlce.rst>
   Lz <orbitlz.rst>
   Op <orbitop.rst>
   Or <orbitor.rst>
   Oz <orbitoz.rst>
   phasedim <orbitphasedim.rst>
   phi <orbitphi.rst>
   pmbb <orbitpmbb.rst>
   pmdec <orbitpmdec.rst>
   pmll <orbitpmll.rst>
   pmra <orbitpmra.rst>
   r <orbitsphr.rst>
   R <orbitr.rst>
   ra <orbitra.rst>
   rap <orbitrap.rst>
   rE <orbitre.rst>
   reshape <orbitreshape.rst>
   rguiding <orbitrguiding.rst>
   rperi <orbitrperi.rst>
   SkyCoord <orbitskycoord.rst>
   SOS <orbitsos.rst>
   theta <orbittheta.rst>
   time <orbittime.rst>
   toLinear <orbittolinear.rst>
   toPlanar <orbittoplanar.rst>
   Tp <orbittp.rst>
   Tr <orbittr.rst>
   TrTp <orbittrtp.rst>
   turn_physical_off <orbitturnphysicaloff.rst>
   turn_physical_on <orbitturnphysicalon.rst>
   Tz <orbittz.rst>
   U <orbitu.rst>
   V <orbitv.rst>
   vbb <orbitvbb.rst>
   vdec <orbitvdec.rst>
   vll <orbitvll.rst>
   vlos <orbitvlos.rst>
   vphi <orbitvphi.rst>
   vr <orbitsphvr.rst>
   vR <orbitvr.rst>
   vra <orbitvra.rst>
   vtheta <orbitvtheta.rst>
   vT <orbitvt.rst>
   vx <orbitvx.rst>
   vy <orbitvy.rst>
   vz <orbitvz.rst>
   W <orbitw.rst>
   wp <orbitwp.rst>
   wr <orbitwr.rst>
   wz <orbitwz.rst>
   x <orbitx.rst>
   y <orbity.rst>
   z <orbitz.rst>
   zmax <orbitzmax.rst>
