.. _bovyconversion:

galpy.util.bovy_conversion
===========================

Utility functions that provide conversions between galpy's *natural*
units and *physical* units. These can be used to translate galpy
outputs in natural coordinates to physical units by multiplying with
the appropriate function. 

These could also be used to figure out the conversion between
different units. For example, if you want to know how many
:math:`\mathrm{GeV\,cm}^{-3}` correspond to
:math:`1\,M_\odot\,\mathrm{pc}^{-3}`, you can calculate

>>> from galpy.util import bovy_conversion
>>> bovy_conversion.dens_in_gevcc(1.,1.)/bovy_conversion.dens_in_msolpc3(1.,1.)
# 37.978342941703616

or :math:`1\,M_\odot\,\mathrm{pc}^{-3} \approx 40\,\mathrm{GeV\,cm}^{-3}`.

Also contains a utility function ``get_physical`` to return the ``ro``
and ``vo`` conversion parameters for any galpy object or lists
thereof.

Functions:
----------

.. toctree::
   :maxdepth: 2

   dens_in_criticaldens <conversiondens_in_criticaldens.rst>
   dens_in_gevcc <conversiondens_in_gevcc.rst>
   dens_in_meanmatterdens <conversiondens_in_meanmatterdens.rst>
   dens_in_msolpc3 <conversiondens_in_msolpc3.rst>
   force_in_2piGmsolpc2 <conversionforce_in_2piGmsolpc2.rst>
   force_in_pcMyr2 <conversionforce_in_pcMyr2.rst>
   force_in_10m13kms2 <conversionforce_in_10m13kms2.rst>
   force_in_kmsMyr <conversionforce_in_kmsMyr.rst>
   freq_in_Gyr <conversionfreq_in_Gyr.rst>
   freq_in_kmskpc <conversionfreq_in_kmskpc.rst>
   get_physical <conversiongetphysical.rst>
   surfdens_in_msolpc2 <conversionsurfdens_in_msolpc2.rst>
   mass_in_msol <conversionmass_in_msol.rst>
   mass_in_1010msol <conversionmass_in_1010msol.rst>
   time_in_Gyr <conversiontime_in_Gyr.rst>
   velocity_in_kpcGyr <conversionvelocity_in_kpcGyr.rst>   
