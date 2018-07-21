###############################################################################
#      class: actionAngleTorus
#
#             Use McMillan, Binney, and Dehnen's Torus code to calculate (x,v)
#             given actions and angles
#
#
###############################################################################
import warnings
import numpy
from galpy.potential import MWPotential, _isNonAxi
from galpy.util import galpyWarning
from . import actionAngleTorus_c
from .actionAngleTorus_c import _ext_loaded as ext_loaded
from galpy.potential.Potential import _check_c
from galpy.potential.Potential import flatten as flatten_potential
_autofit_errvals= {}
_autofit_errvals[-1]= 'something wrong with input, usually bad starting values for the parameters'
_autofit_errvals[-2]= 'Fit failed the goal by a factor <= 2'
_autofit_errvals[-3]= 'Fit failed the goal by more than 2'
_autofit_errvals[-4]= 'Fit aborted: serious problems occured'
class actionAngleTorus(object):
    """Action-angle formalism using the Torus machinery"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleTorus object

        INPUT:

           pot= potential or list of potentials (3D)

           tol= default tolerance to use when fitting tori (|dJ|/J)

           dJ= default action difference when computing derivatives (Hessian or Jacobian)

        OUTPUT:

           instance

        HISTORY:

           2015-08-07 - Written - Bovy (UofT)

        """
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleTorus")
        self._pot= flatten_potential(kwargs['pot'])
        if _isNonAxi(self._pot):
            raise RuntimeError("actionAngleTorus for non-axisymmetric potentials is not supported")
        if self._pot == MWPotential:
            warnings.warn("Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                          galpyWarning)
        if ext_loaded:
            self._c= _check_c(self._pot)
            if not self._c:
                raise RuntimeError('The given potential is not fully implemented in C; using the actionAngleTorus code is not supported in pure Python')
        else:# pragma: no cover
            raise RuntimeError('actionAngleTorus instances cannot be used, because the actionAngleTorus_c extension failed to load')
        self._tol= kwargs.get('tol',0.001)
        self._dJ= kwargs.get('dJ',0.001)
        return None
    
    def __call__(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           [R,vR,vT,z,vz,phi]

        HISTORY:

           2015-08-07 - Written - Bovy (UofT)

        """
        out= actionAngleTorus_c.actionAngleTorus_xvFreqs_c(\
            self._pot,
            jr,jphi,jz,
            angler,anglephi,anglez,
            tol=kwargs.get('tol',self._tol))
        if out[9] != 0:
            warnings.warn("actionAngleTorus' AutoFit exited with non-zero return status %i: %s" % (out[9],_autofit_errvals[out[9]]),
                          galpyWarning)
        return numpy.array(out[:6]).T

    def xvFreqs(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
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

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           ([R,vR,vT,z,vz,phi],OmegaR,Omegaphi,Omegaz,AutoFit error message)

        HISTORY:

           2015-08-07 - Written - Bovy (UofT)

        """
        out= actionAngleTorus_c.actionAngleTorus_xvFreqs_c(\
            self._pot,
            jr,jphi,jz,
            angler,anglephi,anglez,
            tol=kwargs.get('tol',self._tol))
        if out[9] != 0:
            warnings.warn("actionAngleTorus' AutoFit exited with non-zero return status %i: %s" % (out[9],_autofit_errvals[out[9]]),
                          galpyWarning)
        return (numpy.array(out[:6]).T,out[6],out[7],out[8],out[9])

    def Freqs(self,jr,jphi,jz,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequencies corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           (OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2015-08-07 - Written - Bovy (UofT)

        """
        out= actionAngleTorus_c.actionAngleTorus_Freqs_c(\
            self._pot,
            jr,jphi,jz,
            tol=kwargs.get('tol',self._tol))
        if out[3] != 0:
            warnings.warn("actionAngleTorus' AutoFit exited with non-zero return status %i: %s" % (out[3],_autofit_errvals[out[3]]),
                          galpyWarning)
        return out

    def hessianFreqs(self,jr,jphi,jz,**kwargs):
        """
        NAME:

           hessianFreqs

        PURPOSE:

           return the Hessian d Omega / d J and frequencies Omega corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           tol= (object-wide value) goal for |dJ|/|J| along the torus

           dJ= (object-wide value) action difference when computing derivatives (Hessian or Jacobian)

           nosym= (False) if True, don't explicitly symmetrize the Hessian (good to check errors)

        OUTPUT:

           (dO/dJ,Omegar,Omegaphi,Omegaz,Autofit error message)

        HISTORY:

           2016-07-15 - Written - Bovy (UofT)

        """
        out= actionAngleTorus_c.actionAngleTorus_hessian_c(\
            self._pot,
            jr,jphi,jz,
            tol=kwargs.get('tol',self._tol),
            dJ=kwargs.get('dJ',self._dJ))
        if out[4] != 0:
            warnings.warn("actionAngleTorus' AutoFit exited with non-zero return status %i: %s" % (out[4],_autofit_errvals[out[4]]),
                          galpyWarning)
        # Re-arrange frequencies and actions to r,phi,z
        out[0][:,:]= out[0][:,[0,2,1]]
        out[0][:,:]= out[0][[0,2,1]]
        if kwargs.get('nosym',False):
            return out
        else :# explicitly symmetrize
            return (0.5*(out[0]+out[0].T),out[1],out[2],out[3],out[4])

    def xvJacobianFreqs(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           xvJacobianFreqs

        PURPOSE:

           return [R,vR,vT,z,vz,phi], the Jacobian d [R,vR,vT,z,vz,phi] / d (J,angle), the Hessian dO/dJ, and frequencies Omega corresponding to a torus at multiple sets of angles

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           tol= (object-wide value) goal for |dJ|/|J| along the torus

           dJ= (object-wide value) action difference when computing derivatives (Hessian or Jacobian)

           nosym= (False) if True, don't explicitly symmetrize the Hessian (good to check errors)

        OUTPUT:

           ([R,vR,vT,z,vz,phi], [N,6] array

            d[R,vR,vT,z,vz,phi]/d[J,angle], --> (N,6,6) array

            dO/dJ, --> (3,3) array

            Omegar,Omegaphi,Omegaz, [N] arrays

            Autofit error message)

        HISTORY:

           2016-07-19 - Written - Bovy (UofT)

        """
        out= actionAngleTorus_c.actionAngleTorus_jacobian_c(\
            self._pot,
            jr,jphi,jz,
            angler,anglephi,anglez,
            tol=kwargs.get('tol',self._tol),
            dJ=kwargs.get('dJ',self._dJ))
        if out[11] != 0:
            warnings.warn("actionAngleTorus' AutoFit exited with non-zero return status %i: %s" % (out[11],_autofit_errvals[out[11]]),
                          galpyWarning)
        # Re-arrange actions,angles to r,phi,z
        out[6][:,:,:]= out[6][:,:,[0,2,1,3,5,4]]
        out[7][:,:]= out[7][:,[0,2,1]]
        out[7][:,:]= out[7][[0,2,1]]
        # Re-arrange x,v to R,vR,vT,z,vz,phi
        out[6][:,:]= out[6][:,[0,3,5,1,4,2]]
        if not kwargs.get('nosym',False):
            # explicitly symmetrize
            out[7][:]= 0.5*(out[7]+out[7].T)
        return (numpy.array(out[:6]).T,out[6],out[7],
                out[8],out[9],out[10],out[11])
