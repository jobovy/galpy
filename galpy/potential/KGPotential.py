import scipy as sc
from galpy.util import bovy_conversion
from .linearPotential import linearPotential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class KGPotential(linearPotential):
    """Class representing the Kuijken & Gilmore (1989) potential

    .. math::

        \Phi(x) = \\mathrm{amp}\\,\\left(K\\,\\left(\\sqrt{x^2+D^2}-D\\right)+F\\,x^2\\right)

    """
    def __init__(self,K=1.15,F=0.03,D=1.8,amp=1.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a KGPotential

        INPUT:

           K= K parameter (= :math:`2\\pi \\Sigma_{\\mathrm{disk}}`; specify either as surface density or directly as force [i.e., including :math:`2\\pi G`]; can be Quantity)

           F= F parameter (= :math:`4\\pi\\rho_{\\mathrm{halo}}`; specify either as density or directly as second potential derivative [i.e., including :math:`4\\pi G`]; can be Quantity)

           D= D parameter (natural units or Quantity length units)

           amp - an overall amplitude

        OUTPUT:

           instance

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
        linearPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(D,units.Quantity):
            D= D.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(K,units.Quantity):
            try:
                K= K.to(units.pc/units.Myr**2).value\
                    /bovy_conversion.force_in_pcMyr2(self._vo,self._ro)
            except units.UnitConversionError: pass
        if _APY_LOADED and isinstance(K,units.Quantity):
            try:
                K= K.to(units.Msun/units.pc**2).value\
                    /bovy_conversion.force_in_2piGmsolpc2(self._vo,self._ro)
            except units.UnitConversionError:
                raise units.UnitConversionError("Units for K not understood; should be force or surface density")
        if _APY_LOADED and isinstance(F,units.Quantity):
            try:
                F= F.to(units.Msun/units.pc**3).value\
                    /bovy_conversion.dens_in_msolpc3(self._vo,self._ro)*4.*sc.pi
            except units.UnitConversionError: pass
        if _APY_LOADED and isinstance(F,units.Quantity):
            try:
                F= F.to(units.km**2/units.s**2/units.kpc**2).value\
                    *ro**2/vo**2
            except units.UnitConversionError:
                raise units.UnitConversionError("Units for F not understood; should be density")
        self._K= K
        self._F= F
        self._D= D
        self._D2= self._D**2.
        
    def _evaluate(self,x,t=0.):
        return self._K*(sc.sqrt(x**2.+self._D2)-self._D)+self._F*x**2.

    def _force(self,x,t=0.):
        return -x*(self._K/sc.sqrt(x**2+self._D2)+2.*self._F)
