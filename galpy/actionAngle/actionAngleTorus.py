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

from ..potential import MWPotential, _isNonAxi
from ..potential.Potential import _check_c
from ..potential.Potential import flatten as flatten_potential
from ..util import galpyWarning
from . import actionAngleTorus_c
from .actionAngleTorus_c import _ext_loaded as ext_loaded

_autofit_errvals = {}
_autofit_errvals[-1] = (
    "something wrong with input, usually bad starting values for the parameters"
)
_autofit_errvals[-2] = "Fit failed the goal by a factor <= 2"
_autofit_errvals[-3] = "Fit failed the goal by more than 2"
_autofit_errvals[-4] = "Fit aborted: serious problems occurred"


class actionAngleTorus:
    """Action-angle formalism using the Torus machinery"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleTorus object.

        Parameters
        ----------
        pot : potential or list of potentials (3D)
            The potential or list of potentials (3D) to use.
        tol : float, optional
            Default tolerance to use when fitting tori (|dJ|/J).
        dJ : float, optional
            Default action difference when computing derivatives (Hessian or Jacobian).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).


        Notes
        -----
        - 2015-08-07 - Written - Bovy (UofT).
        """
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleTorus")
        self._pot = flatten_potential(kwargs["pot"])
        if _isNonAxi(self._pot):
            raise RuntimeError(
                "actionAngleTorus for non-axisymmetric potentials is not supported"
            )
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if ext_loaded:
            self._c = _check_c(self._pot)
            if not self._c:
                raise RuntimeError(
                    "The given potential is not fully implemented in C; using the actionAngleTorus code is not supported in pure Python"
                )
        else:  # pragma: no cover
            raise RuntimeError(
                "actionAngleTorus instances cannot be used, because the actionAngleTorus_c extension failed to load"
            )
        self._tol = kwargs.get("tol", 0.001)
        self._dJ = kwargs.get("dJ", 0.001)
        return None

    def __call__(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus.

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : numpy.ndarray
            Vertical angle.
        tol : float, optional
            Goal for |dJ|/|J| along the torus. Default is object-wide value.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 6) containing [R, vR, vT, z, vz, phi].

        Notes
        -----
        - 2015-08-07 - Written - Bovy (UofT).
        """
        out = actionAngleTorus_c.actionAngleTorus_xvFreqs_c(
            self._pot,
            jr,
            jphi,
            jz,
            angler,
            anglephi,
            anglez,
            tol=kwargs.get("tol", self._tol),
        )
        if out[9] != 0:
            warnings.warn(
                "actionAngleTorus' AutoFit exited with non-zero return status %i: %s"
                % (out[9], _autofit_errvals[out[9]]),
                galpyWarning,
            )
        return numpy.array(out[:6]).T

    def xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies.

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : arrnumpy.ndarrayay_like
            Vertical angle.
        tol : float, optional
            Goal for |dJ|/|J| along the torus. Default is object-wide value.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - A numpy array of shape (N, 6) containing the phase-space coordinates (R, vR, vT, z, vz, phi) for a number of angles on a single torus.
            - OmegaR : float
                The radial frequency.
            - Omegaphi : float
                The azimuthal frequency.
            - Omegaz : float
                The vertical frequency.
            - AutoFit error message : int
                If AutoFit exited with non-zero return status, a warning message is issued.

        Notes
        -----
        - 2015-08-07 - Written - Bovy (UofT)

        """
        out = actionAngleTorus_c.actionAngleTorus_xvFreqs_c(
            self._pot,
            jr,
            jphi,
            jz,
            angler,
            anglephi,
            anglez,
            tol=kwargs.get("tol", self._tol),
        )
        if out[9] != 0:
            warnings.warn(
                "actionAngleTorus' AutoFit exited with non-zero return status %i: %s"
                % (out[9], _autofit_errvals[out[9]]),
                galpyWarning,
            )
        return (numpy.array(out[:6]).T, out[6], out[7], out[8], out[9])

    def Freqs(self, jr, jphi, jz, **kwargs):
        """
        Return the frequencies corresponding to a torus

        Parameters
        ----------
        jr : float
            Radial action
        jphi : float
            Azimuthal action
        jz : float
            Vertical action
        tol : float, optional
            Goal for |dJ|/|J| along the torus (default is object-wide value)

        Returns
        -------
        tuple
            (OmegaR, Omegaphi, Omegaz)

        Notes
        -----
        - 2015-08-07 - Written - Bovy (UofT)
        """
        out = actionAngleTorus_c.actionAngleTorus_Freqs_c(
            self._pot, jr, jphi, jz, tol=kwargs.get("tol", self._tol)
        )
        if out[3] != 0:
            warnings.warn(
                "actionAngleTorus' AutoFit exited with non-zero return status %i: %s"
                % (out[3], _autofit_errvals[out[3]]),
                galpyWarning,
            )
        return out

    def hessianFreqs(self, jr, jphi, jz, **kwargs):
        """
        Return the Hessian d Omega / d J and frequencies Omega corresponding to a torus

        Parameters
        ----------
        jr : float
            Radial action
        jphi : float
            Azimuthal action
        jz : float
            Vertical action
        tol : float, optional
            Goal for |dJ|/|J| along the torus. Default is object-wide value.
        dJ : float, optional
            Action difference when computing derivatives (Hessian or Jacobian). Default is object-wide value.
        nosym : bool, optional
            If True, don't explicitly symmetrize the Hessian (good to check errors). Default is False.

        Returns
        -------
        tuple
            Tuple containing:
            - dO/dJ
            - Omegar
            - Omegaphi
            - Omegaz
            - Autofit error message

        Notes
        -----
        - 2016-07-15 - Written - Bovy (UofT)
        """
        out = actionAngleTorus_c.actionAngleTorus_hessian_c(
            self._pot,
            jr,
            jphi,
            jz,
            tol=kwargs.get("tol", self._tol),
            dJ=kwargs.get("dJ", self._dJ),
        )
        if out[4] != 0:
            warnings.warn(
                "actionAngleTorus' AutoFit exited with non-zero return status %i: %s"
                % (out[4], _autofit_errvals[out[4]]),
                galpyWarning,
            )
        # Re-arrange frequencies and actions to r,phi,z
        out[0][:, :] = out[0][:, [0, 2, 1]]
        out[0][:, :] = out[0][[0, 2, 1]]
        if kwargs.get("nosym", False):
            return out
        else:  # explicitly symmetrize
            return (0.5 * (out[0] + out[0].T), out[1], out[2], out[3], out[4])

    def xvJacobianFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Return [R,vR,vT,z,vz,phi], the Jacobian d [R,vR,vT,z,vz,phi] / d (J,angle), the Hessian dO/dJ, and frequencies Omega corresponding to a torus at multiple sets of angles

        Parameters
        ----------
        jr : float
            Radial action
        jphi : float
            Azimuthal action
        jz : float
            Vertical action
        angler : numpy.ndarray
            Radial angle
        anglephi : numpy.ndarray
            Azimuthal angle
        anglez : numpy.ndarray
            Vertical angle
        tol : float, optional
            Goal for |dJ|/|J| along the torus (default is object-wide value)
        dJ : float, optional
            Action difference when computing derivatives (Hessian or Jacobian) (default is object-wide value)
        nosym : bool, optional
            If True, don't explicitly symmetrize the Hessian (good to check errors) (default is False)

        Returns
        -------
        tuple
            Tuple containing:
            - ([R,vR,vT,z,vz,phi], [N,6] array
            - d[R,vR,vT,z,vz,phi]/d[J,angle], --> (N,6,6) array
            - dO/dJ, --> (3,3) array
            - Omegar,Omegaphi,Omegaz, [N] arrays
            - Autofit error message)

        Notes
        -----
        - 2016-07-19 - Written - Bovy (UofT)
        """
        out = actionAngleTorus_c.actionAngleTorus_jacobian_c(
            self._pot,
            jr,
            jphi,
            jz,
            angler,
            anglephi,
            anglez,
            tol=kwargs.get("tol", self._tol),
            dJ=kwargs.get("dJ", self._dJ),
        )
        if out[11] != 0:
            warnings.warn(
                "actionAngleTorus' AutoFit exited with non-zero return status %i: %s"
                % (out[11], _autofit_errvals[out[11]]),
                galpyWarning,
            )
        # Re-arrange actions,angles to r,phi,z
        out[6][:, :, :] = out[6][:, :, [0, 2, 1, 3, 5, 4]]
        out[7][:, :] = out[7][:, [0, 2, 1]]
        out[7][:, :] = out[7][[0, 2, 1]]
        # Re-arrange x,v to R,vR,vT,z,vz,phi
        out[6][:, :] = out[6][:, [0, 3, 5, 1, 4, 2]]
        if not kwargs.get("nosym", False):
            # explicitly symmetrize
            out[7][:] = 0.5 * (out[7] + out[7].T)
        return (
            numpy.array(out[:6]).T,
            out[6],
            out[7],
            out[8],
            out[9],
            out[10],
            out[11],
        )
