import math as m
import scipy as sc
from scipy import integrate, stats
from Edf import Edf
from galpy.orbit import Orbit
from galpy.potential_src.linearPotential import evaluatelinearPotentials, evaluatelinearForces
class isothermdf(Edf):
    """An isothermal df f(E) ~ exp(-E/sigma^2)"""
    def __init__(self,normalize=None,pot=None,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an isothermal df
        INPUT:
           sigma=, sigma2=, logsigma=, logsigma2=: either
           normalize= - if True, normalize the df to one, if number, normalize
                        the df to this number (if this is set, pot must be set)
           pot - Potential instance or list of such instances
           +scipy.integrate.quad kwargs for normalization
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        if kwargs.has_key('logsigma2'):
            self._sigma2= m.exp(kwargs['logsigma2'])
            kwargs.pop('logsigma2')
        elif kwargs.has_key['logsigma']:
            self._sigma2= m.exp(kwargs['logsigma']*2.)
            kwargs.pop('logsigma')
        elif kwargs.has_key('sigma'):
            self._sigma2= kwargs['sigma']**2.
            kwargs.pop('sigma')
        elif kwargs.has_key('sigma2'):
            self._sigma2= kwargs['sigma2']
            kwargs.pop('sigma2')
        self._sigma= sc.sqrt(self._sigma2)
        if normalize:
            if pot == None:
                raise AttributeError("if normalize= is set, pot= needs to be set to a Potential as well")
            self._norm= normalize*self._normalize(pot,**kwargs)
        else:
            self._norm= 1.
        return None

    def eval(self,E):
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

    def density(self,x,pot):
        """
        NAME:
           density
        PURPOSE:
           evaluate the DF
        INPUT:
           x - at x
           pot - potential
        OUTPUT:
           rho(x)
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        return self._norm*sc.exp(-evaluatelinearPotentials(x,pot)/self._sigma2)

    def sample(self,pot,n=1):
        """
        NAME:
           sample
        PURPOSE:
           sample from this df
        INPUT:
           pot - potential
           n - number of samples desired
        OUTPUT:
           depending on the dimension, 
           list of [linearOrbit,planarOrbit,Orbit]s
           or a single orbit if n=1
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        BUGS:
           sampling is bad and inaccurate at large x
        """
        #First glean the dimensionality from the potential
        if isinstance(pot,list):
            self._dim= pot[0].dim
        else:
            self._dim= pot.dim
        if self._dim == 1:
            vz= stats.norm.rvs(size=n)*self._sigma 
            #That was easy, now the hard part
            #For now, uniformly sample on [-10.,10.] and reject
            h= self.density(0.,pot)
            zmin, zmax= -10., 10.
            z= []
            while len(z) < n:
                u1= stats.uniform.rvs()*(zmax-zmin)+zmin
                u2= stats.uniform.rvs()*h
                if u2 <= self.density(u1,pot):
                    if m.fabs(u1) > 1.:
                        print u1, "accepted", u2, self.density(u1,pot), h
                    z.append(u1)                
            #Make linearOrbits
            out= []
            for ii in range(n):
                out.append(Orbit(vxvv=[z[ii],vz[ii]]))
            if n == 1:
                return out[0]
            else:
                return out
        elif self._dim == 2:
            raise AttributeError("2 dimensional isothermal distribution functions are not supported at this point")
        elif self._dim == 3:
            raise AttributeError("3 dimensional isothermal distribution functions are not supported at this point")


    def _normalize(self,pot,**kwargs):
        """
        NAME:
           _normalize
        PURPOSE:
           normalize an isothermdf to one
        INPUT:
           pot
           +scipy.integrate.quad kwargs
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        #First glean the dimensionality from the potential
        if isinstance(pot,list):
            self._dim= pot[0].dim
        else:
            self._dim= pot.dim
        #Velocity integration
        out= (2.*m.pi*self._sigma2)**(self._dim/2.)
        if self._dim == 1:
            out/= 2.*integrate.quad(_onedNormalizeIntegrandRaw,
                                    0.,integrate.Inf,
                                    args=(pot,self._sigma2),
                                    **kwargs)[0]
        elif self._dim == 2:
            raise AttributeError("2 dimensional isothermal distribution functions are not supported at this point")
        elif self._dim == 3:
            raise AttributeError("3 dimensional isothermal distribution functions are not supported at this point")
        return out
        
def _onedNormalizeIntegrand(x,pot,sigma2):
    """Internal function that has the normalization integrand for 1D"""
    return 1/x**2.*_onedNormalizeIntegrand(1./x,pot,sigma2)

def _onedNormalizeIntegrandRaw(x,pot,sigma2):
    """Internal function that has the normalization integrand for 1D (untransformed)"""
    return sc.exp(-evaluatelinearPotentials(x,pot)/sigma2)

def _ars_hx_1d(x,args):
    """Internal function that evaluates h(x) for ARS"""
    pot,sigma2= args
    return -evaluatelinearPotentials(x,pot)/sigma2

def _ars_hpx_1d(x,args):
    """Internal function that evaluates h'(x) for ARS"""
    pot,sigma2= args
    return evaluatelinearForces(x,pot)/sigma2
