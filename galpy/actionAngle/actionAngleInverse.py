###############################################################################
# actionAngleInverse.py: top-level class with common routines for inverse
#                        actionAngle methods
###############################################################################
from galpy.util.conversion import actionAngleInverse_physical_input, \
    physical_conversion_actionAngleInverse
from .actionAngle import actionAngle
class actionAngleInverse(actionAngle):
    """actionAngleInverse; top-level class with common routines for inverse actionAngle methods"""
    def __init__(self,*args,**kwargs):
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse('__call__',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           Some sub-classes (like actionAngleTorus) have additional parameters:

              actionAngleTorus:

                  tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           [R,vR,vT,z,vz,phi]

        HISTORY:

           2017-11-14 - Written - Bovy (UofT)

        """
        try:
            return self._evaluate(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'__call__' method not implemented for this actionAngle module")

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse('xvFreqs',pop=True)
    def xvFreqs(self,*args,**kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])


        OUTPUT:

           ([R,vR,vT,z,vz,phi],OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-11-15 - Written - Bovy (UofT)

        """
        try:
            return self._xvFreqs(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'xvFreqs' method not implemented for this actionAngle module")

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse('Freqs',pop=True)
    def Freqs(self,*args,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequencies corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

        OUTPUT:

           (OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-11-15 - Written - Bovy (UofT)

        """
        try:
            return self._Freqs(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'Freqs' method not implemented for this actionAngle module")
