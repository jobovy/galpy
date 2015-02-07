###############################################################################
#   MN3ExponentialDiskPotential.py: class that implements the three Miyamoto-
#                                   Nagai approximation to a radially 
#                                   exponential disk potential of Smith et al.
#                                   2015
###############################################################################
import numpy
from galpy.potential_src.Potential import Potential
class MN3ExponentialDiskPotential(Potential):
    """class that implements the three Miyamoto-Nagai approximation to a radially exponential disk potential of `Smith et al. 2015 <http://adsabs.harvard.edu/abs/2015arXiv150200627S>`_

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R-|z|/h_z\\right)

    or 

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R\\right)\\sech\\left(-|z|/h_z\\right)

    depending on whether sech=True or not. This density is approximated using three Miyamoto-Nagai disks

    """
    def __init__(self,amp=1.,hr=1./3.,hz=1./16.,
                 sech=False,posdens=False,
                 normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a 3MN approximation to an exponential disk potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           hr - disk scale-length

           hz - scale-height

           sech= (False) if True, hz is the scale height of a sech vertical profile (default is exponential vertical profile)

           posdens= (False) if True, allow only positive density solutions (Table 2 in Smith et al. rather than Table 1)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           MN3ExponentialDiskPotential object

        HISTORY:

           2015-02-07 - Written - Bovy (IAS)

        """
        return None
