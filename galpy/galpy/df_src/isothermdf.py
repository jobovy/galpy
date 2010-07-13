import math as m
import scipy as sc
from Edf import Edf
class isothermdf(Edf):
    """An isothermal df f(E) ~ exp(-E/sigma^2"""
    def __init__(**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an isothermal df
        INPUT:
           sigma=, sigma2=, logsigma=, logsigma2=: either
           normalize= - if True, normalize the df to one
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        if kwargs.has_key('logsigma2'):
            self._sigma2= m.exp(kwargs['logsigma2'])
        elif kwargs.has_key['logsigma']:
            self._sigma2= m.exp(kwargs['logsigma']*2.)
        elif kwargs.has_key('sigma'):
            self._sigma2= kwargs['sigma']**2.
        elif kwargs.has_key('sigma2'):
            self._sigma2= kwargs['sigma2']
        self._norm= 1.
        return None

    def eval(E):
        """
        NAME:
           eval
        PURPOSE:
           evaluate the DF
        INPUT:
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        return self._norm*sc.exp(-E/self._sigma2)
