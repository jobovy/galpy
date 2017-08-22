###############################################################################
#   SolidBodyRotationWrapperPotential.py: Wrapper to make a potential rotate
#                                         with a fixed pattern speed, around
#                                         the z axis
###############################################################################
from galpy.potential_src.WrapperPotential import WrapperPotential
from galpy.potential_src.Potential import _APY_LOADED
from galpy.util import bovy_conversion
if _APY_LOADED:
    from astropy import units
class SolidBodyRotationWrapperPotential(WrapperPotential):
    """Potential wrapper class that implements solid-body rotation around the z-axis. Can be used to make a bar or other perturbation rotate. The potential is rotated by replacing 

    .. math::

        \\phi \\rightarrow \\phi + \\Omega \\times t + \\mathrm{pa}

    with :math:`\\Omega` the fixed pattern speed and :math:`\\mathrm{pa}` the position angle at :math:`t=0`.
    """
    normalize= property() # turn off normalize
    def __init__(self,amp=1.,pot=None,omega=1.,pa=0.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a SolidBodyRotationWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; this potential is made to rotate around the z axis by the wrapper

           omega= (1.) the pattern speed (can be a Quantity)

           pa= (0.) the position angle (can be a Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2017-08-22 - Started - Bovy (UofT)

        """
        WrapperPotential.__init__(self,amp=amp,pot=pot,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(omega,units.Quantity):
            omega= omega.to(units.km/units.s/units.kpc).value\
                /bovy_conversion.freq_in_kmskpc(self._vo,self._ro)
        if _APY_LOADED and isinstance(pa,units.Quantity):
            pa= pa.to(units.rad).value
        self._omega= omega
        self._pa= pa
        self.hasC= True
        self.hasC_dxdv= True

    def OmegaP(self):
        """
        NAME:
           OmegaP
        PURPOSE:
           return the pattern speed
        INPUT:
           (none)
        OUTPUT:
           pattern speed
        HISTORY:
           2016-11-02 - Written - Bovy (UofT)
        """
        return self._omega

    def _wrap(self,attribute,R,Z,phi=0.,t=0.):
        return self._wrap_pot_func(attribute)(self._pot,R,Z,t=t,
                                              phi=phi-self._omega*t-self._pa)
