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

# Equations from Table 1
def _mass1_tab1(brd):
    return -0.0090*brd**4.+0.0640*brd**3.-0.1653*brd**2.+0.1164*brd+1.9487
def _mass2_tab1(brd):
    return 0.0173*brd**4.-0.0903*brd**3.+0.0877*brd**2.+0.2029*brd-1.3077
def _mass3_tab1(brd):
    return -0.0051*brd**4.+0.0287*brd**3.-0.0361*brd**2.-0.0544*brd+0.2242
def _a1_tab1(brd):
    return -0.0358*brd**4.+0.2610*brd**3.-0.6987*brd**2.-0.1193*brd+2.0074
def _a2_tab1(brd):
    return -0.0830*brd**4.+0.4992*brd**3.-0.7967*brd**2.-1.2966*brd+4.4441
def _a3_tab1(brd):
    return -0.0247*brd**4.+0.1718*brd**3.-0.4124*brd**2.-0.5944*brd+0.7333
# Equations from Table 2
def _mass1_tab2(brd):
    return 0.0036*brd**4.-0.0330*brd**3.+0.1117*brd**2.-0.1335*brd+0.1749
def _mass2_tab2(brd):
    return -0.0131*brd**4.+0.1090*brd**3.-0.3035*brd**2.+0.2921*brd-5.7976
def _mass3_tab2(brd):
    return -0.0048*brd**4.+0.0454*brd**3.-0.1425*brd**2.+0.1012*brd+6.7120
def _a1_tab2(brd):
    return -0.0158*brd**4.+0.0993*brd**3.-0.2070*brd**2.-0.7089*brd+0.6445
def _a2_tab2(brd):
    return -0.0319*brd**4.+0.1514*brd**3.-0.1279*brd**2.-0.9325*brd+2.6836
def _a3_tab2(brd):
    return -0.0326*brd**4.+0.1816*brd**3.-0.2943*brd**2.-0.6329*brd+2.3193
# Equations to go from hz to b
def _b_exphz(hz):
    return -0.269*hz**3.+1.080*hz**2.+1.092*hz
def _b_sechhz(hz):
    return -0.033*hz**3.+0.262*hz**2.+0.659*hz
