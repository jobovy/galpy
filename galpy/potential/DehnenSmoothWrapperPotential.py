###############################################################################
#   DehnenSmoothWrapperPotential.py: Wrapper to smoothly grow a potential
###############################################################################
from .WrapperPotential import parentWrapperPotential
from .Potential import _APY_LOADED
from galpy.util import bovy_conversion
if _APY_LOADED:
    from astropy import units
class DehnenSmoothWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that implements the growth of a gravitational potential following `Dehnen (2000) <http://adsabs.harvard.edu/abs/2000AJ....119..800D>`__. The amplitude A applied to a potential wrapped by an instance of this class is changed as

    .. math::

        A(t) = amp\\,\\left(\\frac{3}{16}\\xi^5-\\frac{5}{8}\\xi^3+\\frac{15}{16}\\xi+\\frac{1}{2}\\right)

    where

    .. math::

        \\xi = \\begin{cases}
        0 & t < t_\\mathrm{form}\\\\
        2\\left(\\frac{t-t_\\mathrm{form}}{t_\mathrm{steady}}\\right)-1\\,, &  t_\\mathrm{form} \\leq t \\leq t_\\mathrm{form}+t_\\mathrm{steady}\\\\
        1 & t > t_\\mathrm{form}+t_\\mathrm{steady}
        \\end{cases}
    """
    def __init__(self,amp=1.,pot=None,tform=-4.,tsteady=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a DehnenSmoothWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; the amplitude of this will be grown by this wrapper

           tform - start of growth (can be a Quantity)

           tsteady - time from tform at which the potential is fully grown (default: -tform/2, st the perturbation is fully grown at tform/2; can be a Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2017-06-26 - Started - Bovy (UofT)

        """
        if _APY_LOADED and isinstance(tform,units.Quantity):
            tform= tform.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        if _APY_LOADED and isinstance(tsteady,units.Quantity):
            tsteady= tsteady.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        self._tform= tform
        if tsteady is None:
            self._tsteady= self._tform/2.
        else:
            self._tsteady= self._tform+tsteady
        self.hasC= True
        self.hasC_dxdv= True

    def _smooth(self,t):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        return smooth

    def _wrap(self,attribute,*args,**kwargs):
        return self._smooth(kwargs.get('t',0.))\
                *self._wrap_pot_func(attribute)(self._pot,*args,**kwargs)
