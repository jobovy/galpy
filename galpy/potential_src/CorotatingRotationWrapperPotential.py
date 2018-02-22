###############################################################################
#   CorotatingRotationWrapperPotential.py: Wrapper to make a potential rotate
#                                          with a fixed R x pattern speed, 
#                                          around the z axis
###############################################################################
from galpy.potential_src.WrapperPotential import parentWrapperPotential
from galpy.potential_src.Potential import _APY_LOADED
from galpy.util import bovy_conversion
if _APY_LOADED:
    from astropy import units
class CorotatingRotationWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that implements rotation with fixed R x pattern-speed around the z-axis. Can be used to make spiral structure that is everywhere co-rotating. The potential is rotated by replacing 

    .. math::

        \\phi \\rightarrow \\phi + \\frac{V_p(R)}{R} \\times \\left(t-t_0\\right) + \\mathrm{pa}

    with :math:`V_p(R)` the circular velocity curve, :math:`t_0` a reference time---time at which the potential is unchanged by the wrapper---and :math:`\\mathrm{pa}` the position angle at :math:`t=0`. The circular velocity is parameterized as

    .. math::

       V_p(R) = V_{p,0}\,\\left(\\frac{R}{R_0}\\right)^\\beta\,.

    """
    def __init__(self,amp=1.,pot=None,vpo=1.,beta=0.,to=0.,pa=0.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a CorotatingRotationWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; this potential is made to rotate around the z axis by the wrapper

           vpo= (1.) amplitude of the circular-velocity curve (can be a Quantity)

           beta= (0.) power-law amplitude of the circular-velocity curve

           to= (0.) reference time at which the potential == pot

           pa= (0.) the position angle (can be a Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2018-02-21 - Started - Bovy (UofT)

        """
        if _APY_LOADED and isinstance(vpo,units.Quantity):
            vpo= vpo.to(units.km/units.s).value/self._vo
        if _APY_LOADED and isinstance(to,units.Quantity):
            to= to.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        if _APY_LOADED and isinstance(pa,units.Quantity):
            pa= pa.to(units.rad).value
        self._vpo= vpo
        self._beta= beta
        self._pa= pa
        self._to= to
        self.hasC= True
        self.hasC_dxdv= True

    def _wrap(self,attribute,*args,**kwargs):
        kwargs['phi']= kwargs.get('phi',0.)\
            -self._vpo*args[0]**(self._beta-1.)*(kwargs.get('t',0.)-self._to)\
            -self._pa
        return self._wrap_pot_func(attribute)(self._pot,*args,**kwargs)
