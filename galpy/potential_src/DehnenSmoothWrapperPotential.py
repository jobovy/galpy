###############################################################################
#   DehnenSmoothWrapperPotential.py: Wrapper to smoothly grow a potential
###############################################################################
from galpy.potential_src.WrapperPotential import WrapperPotential
class DehnenSmoothWrapperPotential(WrapperPotential):
    normalize= property() # turn off normalize
    def __init__(self,amp=1.,pot=None,tform=-4.,tsteady=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a DehnenSmoothWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; the amplitude of this will be grown by this wrapper

           tform - start of growth

           tsteady - time from tform at which the potential is fully grown (default: -tform/2, st the perturbation is fully grown at tform/2)

        OUTPUT:

           (none)

        HISTORY:

           2017-06-26 - Started - Bovy (UofT)

        """
        WrapperPotential.__init__(self,amp=amp,pot=pot,ro=ro,vo=vo)
        self._tform= tform
        if tsteady is None:
            self._tsteady= self._tform/2.
        else:
            self._tsteady= self._tform+tsteady
        self.hasC= True
        self.hasC_dxdv= False

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

    def _wrap(self,attribute,R,Z,phi=0.,t=0.):
        return self._smooth(t)\
                *self._wrap_pot_func(attribute)(self._pot,R,Z,phi=phi,t=t)
