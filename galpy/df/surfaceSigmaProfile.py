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
import numpy


class surfaceSigmaProfile:
    """Class that contains the surface density and sigma_R^2 profile"""

    def __init__(self):
        """Place holder for implementations of this class"""
        return None

    def formatStringParams(self):
        """
        Return the format-strings to use when writing the parameters of this profile.

        Returns
        -------
        tuple of str
            Tuple of format-strings to use for each parameter in self._params.

        Notes
        -----
        This function defaults to '%6.4f' for each parameter in self._params.

        - 2010-03-28 - Written - Bovy (NYU)
        """
        out = []
        for param in self._params:
            out.append("%6.4f")
        return out

    def outputParams(self):
        """
        Return a tuple of parameters of this profile, to create filenames.

        Returns
        -------
        tuple
            Tuple of parameters (given to % ...).

        Notes
        -----
        - 2010-03-28 - Written - Bovy
        """
        return tuple(self._params)

    def surfacemass(self, R, log=False):
        """
        Return the surface density profile at this R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            Sigma(R)

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU)

        """
        raise NotImplementedError(
            "'surfacemass' function not implemented for this surfaceSigmaProfile class"
        )

    def sigma2(self, R, log=False):
        """
        Return the radial velocity variance at this R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            sigma^2(R).

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU)
        """
        raise NotImplementedError(
            "'sigma2' function not implemented for this surfaceSigmaProfile class"
        )


class expSurfaceSigmaProfile(surfaceSigmaProfile):
    """Exponential surface density and sigma_R^2 class"""

    def __init__(self, params=(1.0 / 3.0, 1.0, 0.2)):
        """
        Initialize an exponential surface density and sigma_R^2 profile.

        Parameters
        ----------
        params : tuple/list
            Tuple/list of (surface scale-length, sigma scale-length, sigma(ro)) (note: *not* sigma2 scale-length)

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU)

        """
        surfaceSigmaProfile.__init__(self)
        self._params = params

    def surfacemass(self, R, log=False):
        """
        Return the surface density profile at this R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        Sigma(R) : float
            Surface density profile at this R.

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU)
        """
        if log:
            return -R / self._params[0]
        else:
            return numpy.exp(-R / self._params[0])

    def surfacemassDerivative(self, R, log=False):
        """
        Return the derivative wrt R of the surface density profile at this R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the derivative of the log (default: False).

        Returns
        -------
        float
            Sigma'(R) or (log Sigma(r) )'.

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU).
        """
        if log:
            return -1.0 / self._params[0]
        else:
            return -numpy.exp(-R / self._params[0]) / self._params[0]

    def sigma2(self, R, log=False):
        """
        Return the radial velocity variance at this R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            sigma^2(R).

        Notes
        -----
        - 2010-03-26 - Written - Bovy (NYU)
        """
        if log:
            return 2.0 * numpy.log(self._params[2]) - 2.0 * (R - 1.0) / self._params[1]
        else:
            return self._params[2] ** 2.0 * numpy.exp(
                -2.0 * (R - 1.0) / self._params[1]
            )

    def sigma2Derivative(self, R, log=False):
        """
        Return the derivative wrt R of the sigma_R^2 profile at this R

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro)
        log : bool, optional
            If True, return the derivative of the log (default: False)

        Returns
        -------
        float
            Sigma_R^2'(R) or (log Sigma_R^2(r) )'

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)
        """

        if log:
            return -2.0 / self._params[1]
        else:
            return (
                self._params[2] ** 2.0
                * numpy.exp(-2.0 * (R - 1.0) / self._params[1])
                * (-2.0 / self._params[1])
            )
