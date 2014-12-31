###############################################################################
#   surfaceSigmaProfile: classes that implement different surface-mass and 
#                        radial velocity dispersion profiles
# 
#   Includes the following:
#      surfaceSigmaProfile - top-level class that represents a surface 
#                            density profile and a sigma_R profile
#      expSurfaceSigmaProfile - class that represents an exponential surface
#                               density profile and an exponential sigma_R 
#                               profile
###############################################################################
import scipy as sc
class surfaceSigmaProfile(object):
    """Class that contains the surface density and sigma_R^2 profile"""
    def __init__(self):
        """Place holder for implementations of this class"""
        return None

    def formatStringParams(self):
        """
        NAME:
           formatStringParams
        PURPOSE:
           when writing the parameters of this profile, what 
           format-strings to use?
           This function defaults to '%6.4f' for each parameter in self._params
        INPUT:
        OUTPUT:
           tuple of format-strings
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        out= []
        for param in self._params:
            out.append('%6.4f')
        return out

    def outputParams(self):
        """
        NAME:
           outputParams
        PURPOSE:'
           return a list of parameters of this profile, to create filenames
        INPUT:
        OUTPUT:
           tuple of parameters (given to % ...)
        HISTORY:
           2010-03-28 - Written - Bovy
        """
        return tuple(self._params)

    def surfacemass(self,R,log=False):
        """
        NAME:
           surfacemass
        PURPOSE:
           return the surface density profile at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the log (default: False)
        OUTPUT:
           Sigma(R)
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        raise NotImplementedError("'surfacemass' function not implemented for this surfaceSigmaProfile class")

    def sigma2(self,R,log=False):
        """
        NAME:
           sigma2
        PURPOSE:
           return the radial velocity variance at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the log (default: False)
        OUTPUT:
           sigma^2(R)
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        raise NotImplementedError("'sigma2' function not implemented for this surfaceSigmaProfile class")
        
class expSurfaceSigmaProfile(surfaceSigmaProfile):
    """Exponential surface density and sigma_R^2 class"""
    def __init__(self,params=(1./3.,1.0,0.2)):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an exponential surface density and sigma_R^2 profile
        INPUT:
           params - tuple/list of (surface scale-length,sigma scale-length,
                    sigma(ro)) (NOTE: *not* sigma2 scale-length)
        OUTPUT:
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        surfaceSigmaProfile.__init__(self)
        self._params= params

    def surfacemass(self,R,log=False):
        """
        NAME:
           surfacemass
        PURPOSE:
           return the surface density profile at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the log (default: False)
        OUTPUT:
           Sigma(R)
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        if log:
            return -R/self._params[0]
        else:
            return sc.exp(-R/self._params[0])

    def surfacemassDerivative(self,R,log=False):
        """
        NAME:
           surfacemassDerivative
        PURPOSE:
           return the derivative wrt R of the surface density profile at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the derivative of the log (default: False)
        OUTPUT:
           Sigma'(R) or (log Sigma(r) )'
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        if log:
            return -1./self._params[0]
        else:
            return -sc.exp(-R/self._params[0])/self._params[0]

    def sigma2(self,R,log=False):
        """
        NAME:
           sigma2
        PURPOSE:
           return the radial velocity variance at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the log (default: False)
        OUTPUT:
           sigma^2(R)
        HISTORY:
           2010-03-26 - Written - Bovy (NYU)
        """
        if log:
            return 2.*sc.log(self._params[2])-2.*(R-1.)/self._params[1]
        else:
            return self._params[2]**2.*sc.exp(-2.*(R-1.)/self._params[1])

    def sigma2Derivative(self,R,log=False):
        """
        NAME:
           sigmaDerivative
        PURPOSE:
           return the derivative wrt R of the sigma_R^2 profile at this R
        INPUT:
           R - Galactocentric radius (/ro)
           log - if True, return the derivative of the log (default: False)
        OUTPUT:
           Sigma_R^2'(R) or (log Sigma_R^2(r) )'
        HISTORY:
           2011-03-24 - Written - Bovy (NYU)
        """
        if log:
            return -2./self._params[1]
        else:
            return self._params[2]**2.*sc.exp(-2.*(R-1.)/self._params[1])\
                *(-2./self._params[1])
