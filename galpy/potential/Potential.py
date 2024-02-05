###############################################################################
#   Potential.py: top-level class for a full potential
#
#   Evaluate by calling the instance: Pot(R,z,phi)
#
#   API for Potentials:
#      function _evaluate(self,R,z,phi) returns Phi(R,z,phi)
#    for orbit integration you need
#      function _Rforce(self,R,z,phi) return -d Phi d R
#      function _zforce(self,R,z,phi) return - d Phi d Z
#    density
#      function _dens(self,R,z,phi) return BOVY??
#    for epicycle frequency
#      function _R2deriv(self,R,z,phi) return d2 Phi dR2
###############################################################################
import os
import os.path
import pickle
from functools import wraps

import numpy
from scipy import integrate, optimize

from ..util import conversion, coords, galpyWarning, plot
from ..util._optional_deps import _APY_LOADED
from ..util.conversion import (
    freq_in_Gyr,
    get_physical,
    physical_conversion,
    potential_physical_input,
    velocity_in_kpcGyr,
)
from .DissipativeForce import DissipativeForce, _isDissipative
from .Force import Force
from .plotEscapecurve import _INF, plotEscapecurve
from .plotRotcurve import plotRotcurve, vcirc

if _APY_LOADED:
    from astropy import units


def check_potential_inputs_not_arrays(func):
    """
    Decorator to check inputs and throw TypeError if any of the inputs are arrays for Potentials that do not support array evaluation.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    function
        Decorated function.

    Notes
    -----
    - 2017-summer - Written for SpiralArmsPotential - Jack Hong (UBC)
    - 2019-05-23 - Moved to Potential for more general use - Bovy (UofT)

    """

    @wraps(func)
    def func_wrapper(self, R, z, phi, t):
        if (
            (hasattr(R, "shape") and R.shape != () and len(R) > 1)
            or (hasattr(z, "shape") and z.shape != () and len(z) > 1)
            or (hasattr(phi, "shape") and phi.shape != () and len(phi) > 1)
            or (hasattr(t, "shape") and t.shape != () and len(t) > 1)
        ):
            raise TypeError(
                f"Methods in {self.__class__.__name__} do not accept array inputs. Please input scalars"
            )
        return func(self, R, z, phi, t)

    return func_wrapper


def potential_positional_arg(func):
    @wraps(func)
    def wrapper(Pot, /, *args, **kwargs):
        return func(Pot, *args, **kwargs)

    return wrapper


class Potential(Force):
    """Top-level class for a potential"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        Initialize a Potential object.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied when evaluating the potential and its forces.
        amp_units : str, optional
            Type of units that `amp` should have if it has units. Possible values are 'mass', 'velocity2', and 'density'.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.

        """
        Force.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        self.dim = 3
        self.isRZ = True
        self.isNonAxi = False
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        return None

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def __call__(self, R, z, phi=0.0, t=0.0, dR=0, dphi=0):
        """
        Evaluate the potential at the specified position and time.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).
        dR : int, optional
            Order of radial derivative (default: 0).
        dphi : int, optional
            Order of azimuthal derivative (default: 0).

        Returns
        -------
        float or Quantity
            The potential at the specified position and time.

        Notes
        -----
        - 2010-04-16 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(R, z, phi=phi, t=t, dR=dR, dphi=dphi)

    def _call_nodecorator(self, R, z, phi=0.0, t=0.0, dR=0.0, dphi=0):
        if dR == 0 and dphi == 0:
            try:
                rawOut = self._evaluate(R, z, phi=phi, t=t)
            except AttributeError:  # pragma: no cover
                raise PotentialError(
                    "'_evaluate' function not implemented for this potential"
                )
            return self._amp * rawOut if not rawOut is None else rawOut
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R, z, phi=phi, t=t, use_physical=False)
        elif dR == 0 and dphi == 1:
            return -self.phitorque(R, z, phi=phi, t=t, use_physical=False)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R, z, phi=phi, t=t, use_physical=False)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R, z, phi=phi, t=t, use_physical=False)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R, z, phi=phi, t=t, use_physical=False)
        elif dR != 0 or dphi != 0:
            raise NotImplementedError(
                "Higher-order derivatives not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rforce(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the cylindrical radial force F_R.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            F_R (R,z,phi,t).

        Notes
        -----
        - 2010-04-16 - Written - Bovy (NYU)

        """
        return self._Rforce_nodecorator(R, z, phi=phi, t=t)

    def _Rforce_nodecorator(self, R, z, phi=0.0, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_Rforce' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def zforce(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the vertical force F_z.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            F_z (R,z,phi,t).

        Notes
        -----
        - 2010-04-16 - Written - Bovy (NYU)

        """
        return self._zforce_nodecorator(R, z, phi=phi, t=t)

    def _zforce_nodecorator(self, R, z, phi=0.0, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._zforce(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_zforce' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def r2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second spherical radial derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2phi/dr2.

        Notes
        -----
        - 2018-03-21 - Written - Webb (UofT)

        """
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self.R2deriv(R, z, phi=phi, t=t, use_physical=False) * R / r
            + self.Rzderiv(R, z, phi=phi, t=t, use_physical=False) * z / r
        ) * R / r + (
            self.Rzderiv(R, z, phi=phi, t=t, use_physical=False) * R / r
            + self.z2deriv(R, z, phi=phi, t=t, use_physical=False) * z / r
        ) * z / r

    @potential_physical_input
    @physical_conversion("density", pop=True)
    def dens(self, R, z, phi=0.0, t=0.0, forcepoisson=False):
        """
        Evaluate the density rho(R,z,t).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).
        forcepoisson : bool, optional
            If True, calculate the density through the Poisson equation, even if an explicit expression for the density exists (default: False).

        Returns
        -------
        float or Quantity
            rho (R,z,phi,t).

        Notes
        -----
        - 2010-08-08 - Written - Bovy (NYU)
        - 2018-03-21 - Modified - Webb (UofT)

        """
        try:
            if forcepoisson:
                raise AttributeError  # Hack!
            return self._amp * self._dens(R, z, phi=phi, t=t)
        except AttributeError:
            # Use the Poisson equation to get the density
            return (
                (
                    -self.Rforce(R, z, phi=phi, t=t, use_physical=False) / R
                    + self.R2deriv(R, z, phi=phi, t=t, use_physical=False)
                    + self.phi2deriv(R, z, phi=phi, t=t, use_physical=False) / R**2.0
                    + self.z2deriv(R, z, phi=phi, t=t, use_physical=False)
                )
                / 4.0
                / numpy.pi
            )

    @potential_physical_input
    @physical_conversion("surfacedensity", pop=True)
    def surfdens(self, R, z, phi=0.0, t=0.0, forcepoisson=False):
        """
        Evaluate the surface density Sigma(R,z,phi,t) = int_{-z}^{+z} dz' rho(R,z',phi,t).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).
        forcepoisson : bool, optional
            If True, calculate the surface density through the Poisson equation, even if an explicit expression for the surface density exists (default: False).

        Returns
        -------
        float or Quantity
            Sigma(R,z,phi,t).

        Notes
        -----
        - 2018-08-19 - Written - Bovy (UofT)
        - 2021-04-19 - Adjusted for non-z-symmetric densities - Bovy (UofT)

        """
        try:
            if forcepoisson:
                raise AttributeError  # Hack!
            return self._amp * self._surfdens(R, z, phi=phi, t=t)
        except AttributeError:
            # Use the Poisson equation to get the surface density
            return (
                (
                    -self.zforce(R, numpy.fabs(z), phi=phi, t=t, use_physical=False)
                    + self.zforce(R, -numpy.fabs(z), phi=phi, t=t, use_physical=False)
                    + integrate.quad(
                        lambda x: -self.Rforce(R, x, phi=phi, t=t, use_physical=False)
                        / R
                        + self.R2deriv(R, x, phi=phi, t=t, use_physical=False)
                        + self.phi2deriv(R, x, phi=phi, t=t, use_physical=False)
                        / R**2.0,
                        -numpy.fabs(z),
                        numpy.fabs(z),
                    )[0]
                )
                / 4.0
                / numpy.pi
            )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the surface density for this potential.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            The surface density.

        Notes
        -----
        - 2018-08-19 - Written - Bovy (UofT).
        - 2021-04-19 - Adjusted for non-z-symmetric densities by Bovy (UofT).

        """
        return integrate.quad(
            lambda x: self._dens(R, x, phi=phi, t=t), -numpy.fabs(z), numpy.fabs(z)
        )[0]

    @potential_physical_input
    @physical_conversion("mass", pop=True)
    def mass(self, R, z=None, t=0.0, forceint=False):
        """
        Evaluate the mass enclosed.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity, optional
            Vertical height up to which to integrate (default: None).
        t : float or Quantity, optional
            Time (default: 0.0).
        forceint : bool, optional
            If True, calculate the mass through integration of the density, even if an explicit expression for the mass exists (default: False).

        Returns
        -------
        float or Quantity
            Mass enclosed within the spherical shell with radius R if z is None else mass in the slab <R and between -z and z; except: potentials inheriting from EllipsoidalPotential, which if z is None return the mass within the ellipsoidal shell with semi-major axis R.

        Notes
        -----
        - 2014-01-29 - Written - Bovy (IAS)
        - 2019-08-15 - Added spherical warning - Bovy (UofT)
        - 2021-03-15 - Changed to integrate to spherical shell for z is None slab otherwise - Bovy (UofT)
        - 2021-03-18 - Switched to using Gauss' theorem - Bovy (UofT)

        """
        from .EllipsoidalPotential import EllipsoidalPotential

        if self.isNonAxi and not isinstance(self, EllipsoidalPotential):
            raise NotImplementedError(
                "mass for non-axisymmetric potentials that are not EllipsoidalPotentials is not currently supported"
            )
        if self.isNonAxi and isinstance(self, EllipsoidalPotential) and not z is None:
            raise NotImplementedError(
                "mass for EllipsoidalPotentials is not currently supported for z != None"
            )
        if not z is None:  # Make sure z is positive, bc we integrate from -z to z
            z = numpy.fabs(z)
        try:
            if forceint:
                raise AttributeError  # Hack!
            return self._amp * self._mass(R, z=z, t=t)
        except AttributeError:
            # Use numerical integration to get the mass, using Gauss' theorem
            if z is None:  # Within spherical shell

                def _integrand(theta):
                    tz = R * numpy.cos(theta)
                    tR = R * numpy.sin(theta)
                    return self.rforce(tR, tz, t=t, use_physical=False) * numpy.sin(
                        theta
                    )

                return -(R**2.0) * integrate.quad(_integrand, 0.0, numpy.pi)[0] / 2.0
            else:  # Within disk at <R, -z --> z
                return (
                    -R
                    * integrate.quad(
                        lambda x: self.Rforce(R, x, t=t, use_physical=False), -z, z
                    )[0]
                    / 2.0
                    - integrate.quad(
                        lambda x: x * self.zforce(x, z, t=t, use_physical=False), 0.0, R
                    )[0]
                )

    @physical_conversion("position", pop=True)
    def rhalf(self, t=0.0, INF=numpy.inf):
        """
        Calculate the half-mass radius, the radius of the spherical shell that contains half the total mass.

        Parameters
        ----------
        t : float or Quantity, optional
            Time (default: 0.0).
        INF : float or Quantity, optional
            Radius at which the total mass is calculated (default: numpy.inf).

        Returns
        -------
        float or Quantity
            Half-mass radius.

        Notes
        -----
        - 2021-03-18 - Written - Bovy (UofT)

        """
        return rhalf(self, t=t, INF=INF, use_physical=False)

    @potential_physical_input
    @physical_conversion("time", pop=True)
    def tdyn(self, R, t=0.0):
        """
        Calculate the dynamical time from tdyn^2 = 3pi/[G<rho>]

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            Dynamical time.

        Notes
        -----
        - 2021-03-18 - Written - Bovy (UofT)

        """
        return 2.0 * numpy.pi * R * numpy.sqrt(R / self.mass(R, use_physical=False))

    @physical_conversion("mass", pop=False)
    def mvir(
        self,
        H=70.0,
        Om=0.3,
        t=0.0,
        overdens=200.0,
        wrtcrit=False,
        forceint=False,
        ro=None,
        vo=None,
        use_physical=False,
    ):  # use_physical necessary bc of pop=False, does nothing inside
        """
        Calculate the virial mass.

        Parameters
        ----------
        H : float, optional
            Hubble constant in km/s/Mpc (default: 70).
        Om : float, optional
            Omega matter (default: 0.3).
        overdens : float, optional
            Overdensity which defines the virial radius (default: 200).
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density (default: False).
        ro : float or Quantity, optional
            Distance scale in kpc (default: object-wide, which if not set is 8 kpc).
        vo : float or Quantity, optional
            Velocity scale in km/s (default: object-wide, which if not set is 220 km/s).
        forceint : bool, optional
            If True, calculate the mass through integration of the density, even if an explicit expression for the mass exists.

        Returns
        -------
        float or Quantity
            M(<rvir).

        Notes
        -----
        - 2014-09-12 - Written - Bovy (IAS)

        """
        if ro is None:
            ro = self._ro
        if vo is None:
            vo = self._vo
        # Evaluate the virial radius
        try:
            rvir = self.rvir(
                H=H,
                Om=Om,
                t=t,
                overdens=overdens,
                wrtcrit=wrtcrit,
                use_physical=False,
                ro=ro,
                vo=vo,
            )
        except AttributeError:
            raise AttributeError(
                "This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius"
            )
        return self.mass(rvir, t=t, forceint=forceint, use_physical=False, ro=ro, vo=vo)

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def R2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Galactocentric azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2phi/dR2.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)

        """
        try:
            return self._amp * self._R2deriv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_R2deriv' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def z2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second vertical derivative.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Galactocentric azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2phi/dz2.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)

        """
        try:
            return self._amp * self._z2deriv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_z2deriv' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def Rzderiv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed R,z derivative.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Galactocentric azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2phi/dz/dR.

        Notes
        -----
        - 2013-08-26 - Written - Bovy (IAS)

        """
        try:
            return self._amp * self._Rzderiv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_Rzderiv' function not implemented for this potential"
            )

    def normalize(self, norm):
        """
        Normalize a potential in such a way that vc(R=1,z=0)=1., or a fraction of this.

        Parameters
        ----------
        norm : float
            Normalize such that Rforce(R=1,z=0) is such that it is 'norm' of the force necessary to make vc(R=1,z=0)=1 (if True, norm=1).

        Returns
        -------
        None

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)

        """
        self._amp *= norm / numpy.fabs(self.Rforce(1.0, 0.0, use_physical=False))

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def phitorque(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the azimuthal torque.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            tau_phi(R, z, phi, t).

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        return self._phitorque_nodecorator(R, z, phi=phi, t=t)

    def _phitorque_nodecorator(self, R, z, phi=0.0, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                raise PotentialError(
                    "'_phitorque' function not implemented for this non-axisymmetric potential"
                )
            return 0.0

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def phi2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second azimuthal derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2Phi/dphi2.

        Notes
        -----
        - 2013-09-24 - Written - Bovy (IAS)
        """
        try:
            return self._amp * self._phi2deriv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                raise PotentialError(
                    "'_phi2deriv' function not implemented for this non-axisymmetric potential"
                )
            return 0.0

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rphideriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed radial, azimuthal derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2Phi/dphidR.

        Notes
        -----
        - 2014-06-30 - Written - Bovy (IAS)
        """
        try:
            return self._amp * self._Rphideriv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                raise PotentialError(
                    "'_Rphideriv' function not implemented for this non-axisymmetric potential"
                )
            return 0.0

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def phizderiv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed azimuthal, vertical derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            d2Phi/dphidz.

        Notes
        -----
        - 2021-04-30 - Written - Bovy (UofT)

        """
        try:
            return self._amp * self._phizderiv(R, z, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                raise PotentialError(
                    "'_phizderiv' function not implemented for this non-axisymmetric potential"
                )
            return 0.0

    def toPlanar(self):
        """
        Convert a 3D potential into a planar potential in the mid-plane.

        Returns
        -------
        planarPotential
        """
        from ..potential import toPlanarPotential

        return toPlanarPotential(self)

    def toVertical(self, R, phi=None, t0=0.0):
        """
        Convert a 3D potential into a linear (vertical) potential at R.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius at which to create the vertical potential.
        phi : float or Quantity, optional
            Galactocentric azimuth at which to create the vertical potential; required for non-axisymmetric potential.
        t0 : float or Quantity, optional
            Time at which to create the vertical potential (default: 0.0)

        Returns
        -------
        linear (vertical) potential : function
            Phi(z,phi,t) = Phi(R,z,phi,t)-Phi(R,0.,phi0,t0) where phi0 and t0 are the phi and t inputs.
        """
        from ..potential import toVerticalPotential

        return toVerticalPotential(self, R, phi=phi, t0=t0)

    def plot(
        self,
        t=0.0,
        rmin=0.0,
        rmax=1.5,
        nrs=21,
        zmin=-0.5,
        zmax=0.5,
        nzs=21,
        effective=False,
        Lz=None,
        phi=None,
        xy=False,
        xrange=None,
        yrange=None,
        justcontours=False,
        levels=None,
        cntrcolors=None,
        ncontours=21,
        savefilename=None,
    ):
        """
        Plot the potential.

        Parameters
        ----------
        t : float, optional
            Time to plot potential at. Default is 0.0.
        rmin : float or Quantity, optional
            Minimum R. Default is 0.0.
        rmax : float or Quantity, optional
            Maximum R. Default is 1.5.
        nrs : int, optional
            Grid in R. Default is 21.
        zmin : float or Quantity, optional
            Minimum z. Default is -0.5.
        zmax : float or Quantity, optional
            Maximum z. Default is 0.5.
        nzs : int, optional
            Grid in z. Default is 21.
        phi : float or Quantity, optional
            Azimuth to use for non-axisymmetric potentials. Default is None.
        xy : bool, optional
            If True, plot the potential in X-Y. Default is False.
        effective : bool, optional
            If True, plot the effective potential Phi + Lz^2/2/R^2. Default is False.
        Lz : float or Quantity, optional
            Angular momentum to use for the effective potential when effective=True. Default is None.
        justcontours : bool, optional
            If True, just plot contours. Default is False.
        savefilename : str, optional
            Save to or restore from this savefile (pickle). Default is None.
        xrange : list, optional
            Can be specified independently from rmin, zmin, etc. Default is None.
        yrange : list, optional
            Can be specified independently from rmin, zmin, etc. Default is None.
        levels : list, optional
            Contours to plot. Default is None.
        cntrcolors : str or list, optional
            Colors of the contours (single color or array with length ncontours). Default is None.
        ncontours : int, optional
            Number of contours when levels is None. Default is 21.

        Returns
        -------
        galpy.util.plot.dens2d return value

        Notes
        -----
        - 2010-07-09 - Written - Bovy (NYU)
        - 2014-04-08 - Added effective= - Bovy (IAS)

        """
        if effective and xy:
            raise RuntimeError("xy and effective cannot be True at the same time")
        rmin = conversion.parse_length(rmin, ro=self._ro)
        rmax = conversion.parse_length(rmax, ro=self._ro)
        zmin = conversion.parse_length(zmin, ro=self._ro)
        zmax = conversion.parse_length(zmax, ro=self._ro)
        Lz = conversion.parse_angmom(Lz, ro=self._ro, vo=self._vo)
        if xrange is None:
            xrange = [rmin, rmax]
        if yrange is None:
            yrange = [zmin, zmax]
        if not savefilename is None and os.path.exists(savefilename):
            print("Restoring savefile " + savefilename + " ...")
            savefile = open(savefilename, "rb")
            potRz = pickle.load(savefile)
            Rs = pickle.load(savefile)
            zs = pickle.load(savefile)
            savefile.close()
        else:
            if effective and Lz is None:
                raise RuntimeError("When effective=True, you need to specify Lz=")
            Rs = numpy.linspace(xrange[0], xrange[1], nrs)
            zs = numpy.linspace(yrange[0], yrange[1], nzs)
            potRz = numpy.zeros((nrs, nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    if xy:
                        R, phi, z = coords.rect_to_cyl(Rs[ii], zs[jj], 0.0)
                    else:
                        R, z = Rs[ii], zs[jj]
                    potRz[ii, jj] = evaluatePotentials(
                        self, R, z, t=t, phi=phi, use_physical=False
                    )
                if effective:
                    potRz[ii, :] += 0.5 * Lz**2 / Rs[ii] ** 2.0
            # Don't plot outside of the desired range
            potRz[Rs < rmin, :] = numpy.nan
            potRz[Rs > rmax, :] = numpy.nan
            potRz[:, zs < zmin] = numpy.nan
            potRz[:, zs > zmax] = numpy.nan
            # Infinity is bad for plotting
            potRz[~numpy.isfinite(potRz)] = numpy.nan
            if not savefilename == None:
                print("Writing savefile " + savefilename + " ...")
                savefile = open(savefilename, "wb")
                pickle.dump(potRz, savefile)
                pickle.dump(Rs, savefile)
                pickle.dump(zs, savefile)
                savefile.close()
        if xy:
            xlabel = r"$x/R_0$"
            ylabel = r"$y/R_0$"
        else:
            xlabel = r"$R/R_0$"
            ylabel = r"$z/R_0$"
        if levels is None:
            levels = numpy.linspace(numpy.nanmin(potRz), numpy.nanmax(potRz), ncontours)
        if cntrcolors is None:
            cntrcolors = "k"
        return plot.dens2d(
            potRz.T,
            origin="lower",
            cmap="gist_gray",
            contours=True,
            xlabel=xlabel,
            ylabel=ylabel,
            xrange=xrange,
            yrange=yrange,
            aspect=0.75 * (rmax - rmin) / (zmax - zmin),
            cntrls="-",
            justcontours=justcontours,
            levels=levels,
            cntrcolors=cntrcolors,
        )

    def plotDensity(
        self,
        t=0.0,
        rmin=0.0,
        rmax=1.5,
        nrs=21,
        zmin=-0.5,
        zmax=0.5,
        nzs=21,
        phi=None,
        xy=False,
        ncontours=21,
        savefilename=None,
        aspect=None,
        log=False,
        justcontours=False,
        **kwargs,
    ):
        """
        Plot the density of this potential.

        Parameters
        ----------
        t : float, optional
            Time to plot potential at.
        rmin : float or Quantity, optional
            Minimum R. If `xy` is True, this is `xmin`.
        rmax : float or Quantity, optional
            Maximum R. If `xy` is True, this is `ymax`.
        nrs : int, optional
            Grid in R.
        zmin : float or Quantity, optional
            Minimum z. If `xy` is True, this is `ymin`.
        zmax : float or Quantity, optional
            Maximum z. If `xy` is True, this is `ymax`.
        nzs : int, optional
            Grid in z.
        phi : float, optional
            Azimuth to use for non-axisymmetric potentials.
        xy : bool, optional
            If True, plot the density in X-Y.
        ncontours : int, optional
            Number of contours.
        justcontours : bool, optional
            If True, just plot contours.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        log : bool, optional
            If True, plot the log density.

        Returns
        -------
        None

        Notes
        -----
        - 2014-01-05 - Written - Bovy (IAS)

        """
        return plotDensities(
            self,
            rmin=rmin,
            rmax=rmax,
            nrs=nrs,
            zmin=zmin,
            zmax=zmax,
            nzs=nzs,
            phi=phi,
            xy=xy,
            t=t,
            ncontours=ncontours,
            savefilename=savefilename,
            justcontours=justcontours,
            aspect=aspect,
            log=log,
            **kwargs,
        )

    def plotSurfaceDensity(
        self,
        t=0.0,
        z=numpy.inf,
        xmin=0.0,
        xmax=1.5,
        nxs=21,
        ymin=-0.5,
        ymax=0.5,
        nys=21,
        ncontours=21,
        savefilename=None,
        aspect=None,
        log=False,
        justcontours=False,
        **kwargs,
    ):
        """
        Plot the surface density of this potential.

        Parameters
        ----------
        t : float, optional
            Time to plot potential at.
        z : float or Quantity, optional
            Height between which to integrate the density (from -z to z).
        xmin : float or Quantity, optional
            Minimum x.
        xmax : float or Quantity, optional
            Maximum x.
        nxs : int, optional
            Grid in x.
        ymin : float or Quantity, optional
            Minimum y.
        ymax : float or Quantity, optional
            Maximum y.
        nys : int, optional
            Grid in y.
        ncontours : int, optional
            Number of contours.
        justcontours : bool, optional
            If True, just plot contours.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        log : bool, optional
            If True, plot the log density.
        **kwargs : dict, optional
            Any additional keyword arguments are passed to `galpy.util.plot.dens2d`.

        Returns
        -------
        None

        Notes
        -----
        - 2020-08-19 - Written - Bovy (UofT)

        """
        return plotSurfaceDensities(
            self,
            xmin=xmin,
            xmax=xmax,
            nxs=nxs,
            ymin=ymin,
            ymax=ymax,
            nys=nys,
            t=t,
            z=z,
            ncontours=ncontours,
            savefilename=savefilename,
            justcontours=justcontours,
            aspect=aspect,
            log=log,
            **kwargs,
        )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def vcirc(self, R, phi=None, t=0.0):
        """
        Calculate the circular velocity at R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        phi : float, optional
            Azimuth to use for non-axisymmetric potentials.
        t : float or Quantity, optional
            Time.

        Returns
        -------
        float or Quantity
            Circular rotation velocity.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)
        - 2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return numpy.sqrt(R * -self.Rforce(R, 0.0, phi=phi, t=t, use_physical=False))

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def dvcircdR(self, R, phi=None, t=0.0):
        """
        Calculate the derivative of the circular velocity at R with respect to R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        phi : float, optional
            Azimuth to use for non-axisymmetric potentials.
        t : float or Quantity, optional
            Time. Default: 0.0

        Returns
        -------
        float or Quantity
            Derivative of the circular rotation velocity with respect to R.

        Notes
        -----
        - 2013-01-08 - Written - Bovy (IAS)
        - 2016-06-28 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return (
            0.5
            * (
                -self.Rforce(R, 0.0, phi=phi, t=t, use_physical=False)
                + R * self.R2deriv(R, 0.0, phi=phi, t=t, use_physical=False)
            )
            / self.vcirc(R, phi=phi, t=t, use_physical=False)
        )

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def omegac(self, R, t=0.0):
        """
        Calculate the circular angular speed at R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time. Default: 0.0

        Returns
        -------
        float or Quantity
            Circular angular speed.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)

        """
        return numpy.sqrt(-self.Rforce(R, 0.0, t=t, use_physical=False) / R)

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def epifreq(self, R, t=0.0):
        """
        Calculate the epicycle frequency at R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time. Default: 0.0

        Returns
        -------
        float or Quantity
            Epicycle frequency.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)

        """
        return numpy.sqrt(
            self.R2deriv(R, 0.0, t=t, use_physical=False)
            - 3.0 / R * self.Rforce(R, 0.0, t=t, use_physical=False)
        )

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def verticalfreq(self, R, t=0.0):
        """
        Calculate the vertical frequency at R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time. Default: 0.0

        Returns
        -------
        float or Quantity
            Vertical frequency.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)

        """
        return numpy.sqrt(self.z2deriv(R, 0.0, t=t, use_physical=False))

    @physical_conversion("position", pop=True)
    def lindbladR(self, OmegaP, m=2, t=0.0, **kwargs):
        """
        Calculate the radius of a Lindblad resonance.

        Parameters
        ----------
        OmegaP : float or Quantity
            Pattern speed.
        m : int or str, optional
            Order of the resonance (as in m(O-Op)=kappa (negative m for outer)).
            Use m='corotation' for corotation.
            Default: 2.
        t : float or Quantity, optional
            Time. Default: 0.0.
        **kwargs: dict, optional
            Additional parameters passed to scipy.optimize.brentq.

        Returns
        -------
        float or Quantity or None
            Radius of Lindblad resonance. None if there is no resonance.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)

        """
        OmegaP = conversion.parse_frequency(OmegaP, ro=self._ro, vo=self._vo)
        return lindbladR(self, OmegaP, m=m, t=t, use_physical=False, **kwargs)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def vesc(self, R, t=0.0):
        """
        Calculate the escape velocity at R for this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time. Default: 0.0.

        Returns
        -------
        float or Quantity
            Escape velocity.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)

        """
        return numpy.sqrt(
            2.0
            * (
                self(_INF, 0.0, t=t, use_physical=False)
                - self(R, 0.0, t=t, use_physical=False)
            )
        )

    @physical_conversion("position", pop=True)
    def rl(self, lz, t=0.0):
        """
        Calculate the radius of a circular orbit of Lz.

        Parameters
        ----------
        lz : float or Quantity
            Angular momentum.
        t : float or Quantity, optional
            Time. Default: 0.0.

        Returns
        -------
        float or Quantity
            Radius.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)
        - An efficient way to call this function on many objects is provided as the Orbit method rguiding.

        See Also
        --------
        galpy.orbit.Orbit.rguiding
        """
        lz = conversion.parse_angmom(lz, ro=self._ro, vo=self._vo)
        return rl(self, lz, t=t, use_physical=False)

    @physical_conversion("position", pop=True)
    def rE(self, E, t=0.0):
        """
        Calculate the radius of a circular orbit with energy E.

        Parameters
        ----------
        E : float or Quantity
            Energy.
        t : float or Quantity, optional
            Time. Default: 0.0.

        Returns
        -------
        float or Quantity
            Radius.

        Notes
        -----
        - 2022-04-06 - Written - Bovy (UofT)
        - An efficient way to call this function on many objects is provided as the Orbit method rE.

        See Also
        --------
        galpy.orbit.Orbit.rE
        """
        E = conversion.parse_energy(E, ro=self._ro, vo=self._vo)
        return rE(self, E, t=t, use_physical=False)

    @physical_conversion("action", pop=True)
    def LcE(self, E, t=0.0):
        """
        Calculate the angular momentum of a circular orbit with energy E.

        Parameters
        ----------
        E : float or Quantity
            Energy.
        t : float or Quantity, optional
            Time. Default: 0.0.

        Returns
        -------
        float or Quantity
            Lc(E).

        Notes
        -----
        - 2022-04-06 - Written - Bovy (UofT).

        """
        E = conversion.parse_energy(E, ro=self._ro, vo=self._vo)
        return LcE(self, E, t=t, use_physical=False)

    @potential_physical_input
    @physical_conversion("dimensionless", pop=True)
    def flattening(self, R, z, t=0.0):
        """
        Calculate the potential flattening, defined as sqrt(fabs(z/R F_R/F_z))

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity
            Height.
        t : float or Quantity, optional
            Time. Default: 0.0.

        Returns
        -------
        float or Quantity
            Flattening.

        Notes
        -----
        - 2012-09-13 - Written - Bovy (IAS)

        """
        return numpy.sqrt(
            numpy.fabs(
                z
                / R
                * self.Rforce(R, z, t=t, use_physical=False)
                / self.zforce(R, z, t=t, use_physical=False)
            )
        )

    @physical_conversion("velocity", pop=True)
    def vterm(self, l, t=0.0, deg=True):
        """
        Calculate the terminal velocity at l in this potential.

        Parameters
        ----------
        l : float or Quantity
            Galactic longitude [deg/rad; can be Quantity).
        t : float or Quantity, optional
            Time. Default: 0.0.
        deg : bool, optional
            If True (default), l in deg.

        Returns
        -------
        float or Quantity
            Terminal velocity.

        Notes
        -----
        - 2013-05-31 - Written - Bovy (IAS).

        """
        if _APY_LOADED and isinstance(l, units.Quantity):
            l = conversion.parse_angle(l)
            deg = False
        if deg:
            sinl = numpy.sin(l / 180.0 * numpy.pi)
        else:
            sinl = numpy.sin(l)
        return sinl * (
            self.omegac(numpy.fabs(sinl), t=t, use_physical=False)
            - self.omegac(1.0, t=t, use_physical=False)
        )

    def plotRotcurve(self, *args, **kwargs):
        """
        Plot the rotation curve for this potential (in the z=0 plane for non-spherical potentials).

        Parameters
        ----------
        Rrange : float or Quantity, optional
            Range to plot. Default: [0.01, 2.] * ro.
        grid : int, optional
            Number of points to plot. Default: 1001.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        *args, **kwargs :
            Arguments and keyword arguments for galpy.util.plot.plot.

        Returns
        -------
        matplotlib.pyplot.axis
            Plot to output device.

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        return plotRotcurve(self, *args, **kwargs)

    def plotEscapecurve(self, *args, **kwargs):
        """
        Plot the escape velocity curve for this potential (in the z=0 plane for non-spherical potentials).

        Parameters
        ----------
        Rrange : float or Quantity, optional
            Range to plot. Default: [0.01, 2.] * ro.
        grid : int, optional
            Number of points to plot. Default: 1001.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        *args, **kwargs :
            Arguments and keyword arguments for galpy.util.plot.plot.

        Returns
        -------
        matplotlib.pyplot.axis
            Plot to output device.

        Notes
        -----
        - 2010-08-08 - Written - Bovy (NYU).

        """
        return plotEscapecurve(self.toPlanar(), *args, **kwargs)

    def conc(
        self, H=70.0, Om=0.3, t=0.0, overdens=200.0, wrtcrit=False, ro=None, vo=None
    ):
        """
        Return the concentration.

        Parameters
        ----------
        H : float, optional
            Hubble constant in km/s/Mpc. Default: 70.0.
        Om : float, optional
            Omega matter. Default: 0.3.
        t : float or Quantity, optional
            Time. Default: 0.0.
        overdens : float, optional
            Overdensity which defines the virial radius. Default: 200.0.
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density. Default: False.
        ro : float or Quantity, optional
            Distance scale in kpc. Default: object-wide, which if not set is 8 kpc.
        vo : float or Quantity, optional
            Velocity scale in km/s. Default: object-wide, which if not set is 220 km/s.

        Returns
        -------
        float
            Concentration (scale/rvir).

        Notes
        -----
        - 2014-04-03 - Written - Bovy (IAS)

        """
        if ro is None:
            ro = self._ro
        if vo is None:
            vo = self._vo
        try:
            return (
                self.rvir(
                    H=H,
                    Om=Om,
                    t=t,
                    overdens=overdens,
                    wrtcrit=wrtcrit,
                    ro=ro,
                    vo=vo,
                    use_physical=False,
                )
                / self._scale
            )
        except AttributeError:
            raise AttributeError(
                "This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius"
            )

    def nemo_accname(self):
        """
        Return the accname potential name for use of this potential with NEMO.

        Returns
        -------
        str
            Acceleration name.

        Notes
        -----
        - 2014-12-18 - Written - Bovy (IAS)

        """
        try:
            return self._nemo_accname
        except AttributeError:
            raise AttributeError(
                "NEMO acceleration name not supported for %s" % self.__class__.__name__
            )

    def nemo_accpars(self, vo, ro):
        """
        Return the accpars potential parameters for use of this potential with NEMO.

        Parameters
        ----------
        vo : float
            Velocity unit in km/s.
        ro : float
            Length unit in kpc.

        Returns
        -------
        str
            Accpars string.

        Notes
        -----
        - 2014-12-18 - Written - Bovy (IAS)

        """
        try:
            return self._nemo_accpars(vo, ro)
        except AttributeError:
            raise AttributeError(
                "NEMO acceleration parameters not supported for %s"
                % self.__class__.__name__
            )

    @potential_physical_input
    @physical_conversion("position", pop=True)
    def rtide(self, R, z, phi=0.0, t=0.0, M=None):
        """
        Calculate the tidal radius for object of mass M assuming a circular orbit

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius
        z : float or Quantity
            height
        phi : float or Quantity, optional
            azimuth (default: 0.0)
        t : float or Quantity, optional
            time (default: 0.0)
        M : float or Quantity
            mass of the object

        Returns
        -------
        float or Quantity
            Tidal radius

        Notes
        -----
        - 2018-03-21 - Written - Webb (UofT)
        - The tidal radius is computed as

           .. math::

               r_t^3 = \\frac{GM_s}{\\Omega^2-\\mathrm{d}^2\\Phi/\\mathrm{d}r^2}

          where :math:`M_s` is the cluster mass, :math:`\\Omega` is the circular frequency, and :math:`\\Phi` is the gravitational potential. For non-spherical potentials, we evaluate :math:`\\Omega^2 = (1/r)(\\mathrm{d}\\Phi/\\mathrm{d}r)` and evaluate the derivatives at the given position of the cluster.
        """

        if M is None:
            # Make sure an object mass is given
            raise PotentialError(
                "Mass parameter M= needs to be set to compute tidal radius"
            )
        r = numpy.sqrt(R**2.0 + z**2.0)
        omegac2 = -self.rforce(R, z, phi=phi, t=t, use_physical=False) / r
        d2phidr2 = self.r2deriv(R, z, phi=phi, t=t, use_physical=False)
        return (M / (omegac2 - d2phidr2)) ** (1.0 / 3.0)

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def ttensor(self, R, z, phi=0.0, t=0.0, eigenval=False):
        """
        Calculate the tidal tensor Tij=-d(Psi)(dxidxj)

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius
        z : float or Quantity
            height
        phi : float or Quantity, optional
            azimuth (default: 0.0)
        t : float or Quantity, optional
            time (default: 0.0)
        eigenval : bool, optional
            return eigenvalues if true (default: False)

        Returns
        -------
        Tidal Tensor

        Notes
        -----
        - 2018-03-21 - Written - Webb (UofT)

        """
        if self.isNonAxi:
            raise PotentialError(
                "Tidal tensor calculation is currently only implemented for axisymmetric potentials"
            )
        # Evaluate forces, angles and derivatives
        Rderiv = -self.Rforce(R, z, phi=phi, t=t, use_physical=False)
        phideriv = -self.phitorque(R, z, phi=phi, t=t, use_physical=False)
        R2deriv = self.R2deriv(R, z, phi=phi, t=t, use_physical=False)
        z2deriv = self.z2deriv(R, z, phi=phi, t=t, use_physical=False)
        phi2deriv = self.phi2deriv(R, z, phi=phi, t=t, use_physical=False)
        Rzderiv = self.Rzderiv(R, z, phi=phi, t=t, use_physical=False)
        Rphideriv = self.Rphideriv(R, z, phi=phi, t=t, use_physical=False)
        # Temporarily set zphideriv to zero until zphideriv is added to Class
        zphideriv = 0.0
        cosphi = numpy.cos(phi)
        sinphi = numpy.sin(phi)
        cos2phi = cosphi**2.0
        sin2phi = sinphi**2.0
        R2 = R**2.0
        R3 = R**3.0
        # Tidal tensor
        txx = (
            R2deriv * cos2phi
            - Rphideriv * 2.0 * cosphi * sinphi / R
            + Rderiv * sin2phi / R
            + phi2deriv * sin2phi / R2
            + phideriv * 2.0 * cosphi * sinphi / R2
        )
        tyx = (
            R2deriv * sinphi * cosphi
            + Rphideriv * (cos2phi - sin2phi) / R
            - Rderiv * sinphi * cosphi / R
            - phi2deriv * sinphi * cosphi / R2
            + phideriv * (sin2phi - cos2phi) / R2
        )
        tzx = Rzderiv * cosphi - zphideriv * sinphi / R
        tyy = (
            R2deriv * sin2phi
            + Rphideriv * 2.0 * cosphi * sinphi / R
            + Rderiv * cos2phi / R
            + phi2deriv * cos2phi / R2
            - phideriv * 2.0 * sinphi * cosphi / R2
        )
        txy = tyx
        tzy = Rzderiv * sinphi + zphideriv * cosphi / R
        txz = tzx
        tyz = tzy
        tzz = z2deriv
        tij = -numpy.array([[txx, txy, txz], [tyx, tyy, tyz], [tzx, tzy, tzz]])
        if eigenval:
            return numpy.linalg.eigvals(tij)
        else:
            return tij

    @physical_conversion("position", pop=True)
    def zvc(self, R, E, Lz, phi=0.0, t=0.0):
        """
        Calculate the zero-velocity curve.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius
        E : float or Quantity
            Energy
        Lz : float or Quantity
            Angular momentum
        phi : float or Quantity, optional
            azimuth (default: 0.0)
        t : float or Quantity, optional
            time (default: 0.0)

        Returns
        -------
        z : float or Quantity
            z such that Phi(R,z) + Lz/[2R^2] = E

        Notes
        -----
        - 2020-08-20 - Written - Bovy (UofT)

        """
        return zvc(self, R, E, Lz, phi=phi, t=t, use_physical=False)

    @physical_conversion("position", pop=True)
    def zvc_range(self, E, Lz, phi=0.0, t=0.0):
        """
        Calculate the minimum and maximum radius for which the zero-velocity curve exists for this energy and angular momentum (R such that Phi(R,0) + Lz/[2R^2] = E)

        Parameters
        ----------
        E : float or Quantity
            Energy
        Lz : float or Quantity
            Angular momentum
        phi : float or Quantity, optional
            azimuth (default: 0.0)
        t : float or Quantity, optional
            time (default: 0.0)

        Returns
        -------
        Rmin, Rmax : float or Quantity
            Solutions R such that Phi(R,0) + Lz/[2R^2] = E

        Notes
        -----
        - 2020-08-20 - Written - Bovy (UofT)
        """
        return zvc_range(self, E, Lz, phi=phi, t=t, use_physical=False)


class PotentialError(Exception):  # pragma: no cover
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluatePotentials(Pot, R, z, phi=None, t=0.0, dR=0, dphi=0):
    """
    Evaluate a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential
        Potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    dR : int, optional
        If set to a non-zero integer, return the dR derivative instead (default: 0).
    dphi : int, optional
        If set to a non-zero integer, return the dphi derivative instead (default: 0).

    Returns
    -------
    float or Quantity
        Potential or potential derivative.

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU)

    """
    return _evaluatePotentials(Pot, R, z, phi=phi, t=t, dR=dR, dphi=dphi)


def _evaluatePotentials(Pot, R, z, phi=None, t=0.0, dR=0, dphi=0):
    """Raw, undecorated function for internal use"""
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    isList = isinstance(Pot, list)
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot._call_nodecorator(R, z, phi=phi, t=t, dR=dR, dphi=dphi)
        return out
    elif isinstance(Pot, Potential):
        return Pot._call_nodecorator(R, z, phi=phi, t=t, dR=dR, dphi=dphi)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("density", pop=True)
def evaluateDensities(Pot, R, z, phi=None, t=0.0, forcepoisson=False):
    """
    Evaluate the density corresponding to a potential or sum of potentials.

    Parameters
    ----------
    Pot : potential or list of potentials
        Dissipative forces in such a list are ignored.
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    forcepoisson : bool, optional
        If True, calculate the density through the Poisson equation, even if an explicit expression for the density exists.

    Returns
    -------
    float or Quantity
        density

    Notes
    -----
    - 2010-08-08 - Written - Bovy (NYU)
    - 2013-12-28 - Added forcepoisson - Bovy (IAS)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.dens(
                    R, z, phi=phi, t=t, forcepoisson=forcepoisson, use_physical=False
                )
        return out
    elif isinstance(Pot, Potential):
        return Pot.dens(
            R, z, phi=phi, t=t, forcepoisson=forcepoisson, use_physical=False
        )
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateDensities' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("surfacedensity", pop=True)
def evaluateSurfaceDensities(Pot, R, z, phi=None, t=0.0, forcepoisson=False):
    """
    Evaluate the surface density for a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential
        Potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    forcepoisson : bool, optional
        If True, calculate the surface density through the Poisson equation, even if an explicit expression for the surface density exists.

    Returns
    -------
    float or Quantity
        Surface density.

    Notes
    -----
    - 2018-08-20 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.surfdens(
                    R, z, phi=phi, t=t, forcepoisson=forcepoisson, use_physical=False
                )
        return out
    elif isinstance(Pot, Potential):
        return Pot.surfdens(
            R, z, phi=phi, t=t, forcepoisson=forcepoisson, use_physical=False
        )
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateSurfaceDensities' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("mass", pop=True)
def mass(Pot, R, z=None, t=0.0, forceint=False):
    """
    Calculate the mass enclosed either within a spherical shell with radius R or in the slab <R and between -z and z.

    Parameters
    ----------
    Pot : Potential or list of Potentials
        Potential or list of Potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity, optional
        Vertical height up to which to integrate. Default is None.
    t : float or Quantity, optional
        Time. Default: 0.0.
    forceint : bool, optional
        If True, calculate the mass through integration of the density, even if an explicit expression for the mass exists. Default is False.

    Returns
    -------
    float or Quantity
        Mass enclosed within the spherical shell with radius R if z is None else mass in the slab <R and between -z and z.

    Notes
    -----
    - 2021-02-07 - Written - Bovy (UofT)
    - 2021-03-15 - Changed to integrate to spherical shell for z is None slab otherwise - Bovy (UofT)

    """
    Pot = flatten(Pot)
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi:
        raise NotImplementedError(
            "mass for non-axisymmetric potentials is not currently supported"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.mass(R, z=z, t=t, forceint=forceint, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.mass(R, z=z, t=t, forceint=forceint, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'mass' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluateRforces(Pot, R, z, phi=None, t=0.0, v=None):
    """
    Evaluate the radial force F_R(R,z,phi,t) of a potential, force or a list of potentials/forces.

    Parameters
    ----------
    Pot : Potential, DissipativeForce or list of Potential or DissipativeForce instances
        A potential, dissipative force or a list of such objects.
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates. Required when including dissipative forces. Default is None.

    Returns
    -------
    F_R : float or Quantity
        Radial force F_R(R,z,phi,t).

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU)
    - 2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluateRforces(Pot, R, z, phi=phi, t=t, v=v)


def _evaluateRforces(Pot, R, z, phi=None, t=0.0, v=None):
    """Raw, undecorated function for internal use"""
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) Potential instances includes dissipative components, but you did not provide the 3D velocity (required for dissipative forces)"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if isinstance(pot, DissipativeForce):
                out += pot._Rforce_nodecorator(R, z, phi=phi, t=t, v=v)
            else:
                out += pot._Rforce_nodecorator(R, z, phi=phi, t=t)
        return out
    elif isinstance(Pot, Potential):
        return Pot._Rforce_nodecorator(R, z, phi=phi, t=t)
    elif isinstance(Pot, DissipativeForce):
        return Pot._Rforce_nodecorator(R, z, phi=phi, t=t, v=v)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateRforces' is neither a Potential-instance, DissipativeForce-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluatephitorques(Pot, R, z, phi=None, t=0.0, v=None):
    """
    Evaluate the azimuthal torque due to a potential, force or a list of potentials/forces.

    Parameters
    ----------
    Pot : Potential, DissipativeForce or list of Potential or DissipativeForce instances
        A potential, dissipative force or a list of such objects.
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray, optional
        Current velocity in cylindrical coordinates. Required when including dissipative forces. Default is None.

    Returns
    -------
    float or Quantity
        The azimuthal torque.

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU)
    - 2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluatephitorques(Pot, R, z, phi=phi, t=t, v=v)


def _evaluatephitorques(Pot, R, z, phi=None, t=0.0, v=None):
    """Raw, undecorated function for internal use"""
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if isinstance(pot, DissipativeForce):
                out += pot._phitorque_nodecorator(R, z, phi=phi, t=t, v=v)
            else:
                out += pot._phitorque_nodecorator(R, z, phi=phi, t=t)
        return out
    elif isinstance(Pot, Potential):
        return Pot._phitorque_nodecorator(R, z, phi=phi, t=t)
    elif isinstance(Pot, DissipativeForce):
        return Pot._phitorque_nodecorator(R, z, phi=phi, t=t, v=v)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatephitorques' is neither a Potential-instance, DissipativeForce-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluatezforces(Pot, R, z, phi=None, t=0.0, v=None):
    """
    Evaluate the vertical force at a given position due to a potential, force or a list of potentials/forces.

    Parameters
    ----------
    Pot : Potential, DissipativeForce or list of Potential or DissipativeForce instances
        A potential, dissipative force or a list of such objects.
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates. Required when including dissipative forces. Default is None.

    Returns
    -------
    float or Quantity
        The vertical force F_z(R,z,phi,t).

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU)
    - 2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluatezforces(Pot, R, z, phi=phi, t=t, v=v)


def _evaluatezforces(Pot, R, z, phi=None, t=0.0, v=None):
    """Raw, undecorated function for internal use"""
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if isinstance(pot, DissipativeForce):
                out += pot._zforce_nodecorator(R, z, phi=phi, t=t, v=v)
            else:
                out += pot._zforce_nodecorator(R, z, phi=phi, t=t)
        return out
    elif isinstance(Pot, Potential):
        return Pot._zforce_nodecorator(R, z, phi=phi, t=t)
    elif isinstance(Pot, DissipativeForce):
        return Pot._zforce_nodecorator(R, z, phi=phi, t=t, v=v)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatezforces' is neither a Potential-instance, DissipativeForce-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluaterforces(Pot, R, z, phi=None, t=0.0, v=None):
    """
    Evaluate the radial force at a given position due to a potential, force or a list of potentials/forces.

    Parameters
    ----------
    Pot : Potential, DissipativeForce or list of Potential or DissipativeForce instances
        A potential, dissipative force or a list of such objects.
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates. Required when including dissipative forces. Default is None.

    Returns
    -------
    float or Quantity
        The radial force F_r(R,z,phi,t).

    Notes
    -----
    - 2016-06-10 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if isinstance(pot, DissipativeForce):
                out += pot.rforce(R, z, phi=phi, t=t, v=v, use_physical=False)
            else:
                out += pot.rforce(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.rforce(R, z, phi=phi, t=t, use_physical=False)
    elif isinstance(Pot, DissipativeForce):
        return Pot.rforce(R, z, phi=phi, t=t, v=v, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluaterforces' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def evaluateR2derivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second (cylindrical) radial derivative of a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second (cylindrical) radial derivative d2Phi/d2R of the potential.

    Notes
    -----
    - 2012-07-25 - Written - Bovy (IAS)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.R2deriv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.R2deriv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateR2derivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def evaluatez2derivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second vertical derivative of a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second vertical derivative d2Phi/d2z of the potential.

    Notes
    -----
    - 2012-07-25 - Written - Bovy (IAS)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.z2deriv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.z2deriv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatez2derivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def evaluateRzderivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second derivative of the sum of potentials with respect to cylindrical Galactocentric distance and height.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second derivative d2Phi/dRdz of the sum of potentials with respect to cylindrical Galactocentric distance and height.

    Notes
    -----
    - 2013-08-28 - Written - Bovy (IAS)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.Rzderiv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.Rzderiv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateRzderivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluatephi2derivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second azimuthal derivative of a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second azimuthal derivative d2Phi/d2phi of the potential.

    Notes
    -----
    - 2018-03-28 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.phi2deriv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.phi2deriv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatephi2derivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluateRphiderivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second derivative of the sum of potentials with respect to cylindrical Galactocentric distance and azimuth.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second derivative d2Phi/dRdphi of the sum of potentials with respect to cylindrical Galactocentric distance and azimuth.

    Notes
    -----
    - 2014-06-30 - Written - Bovy (IAS)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.Rphideriv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.Rphideriv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateRphiderivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluatephizderivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second derivative of the sum of potentials with respect to cylindrical azimuth and height.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second derivative d2Phi/dphidz of the sum of potentials with respect to cylindrical azimuth and height.

    Notes
    -----
    - 2021-04-30 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.phizderiv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.phizderiv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatephizderivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def evaluater2derivs(Pot, R, z, phi=None, t=0.0):
    """
    Evaluate the second (spherical) radial derivative of a potential or sum of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        A potential or list of potentials (dissipative forces in such a list are ignored).
    R : float or Quantity
        Cylindrical Galactocentric distance.
    z : float or Quantity
        Distance above the plane.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        The second (spherical) radial derivative d2Phi/d2r  of the potential.

    Notes
    -----
    - 2018-03-28 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) Potential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList:
        out = 0.0
        for pot in Pot:
            if not isinstance(pot, DissipativeForce):
                out += pot.r2deriv(R, z, phi=phi, t=t, use_physical=False)
        return out
    elif isinstance(Pot, Potential):
        return Pot.r2deriv(R, z, phi=phi, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluater2derivs' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
def plotPotentials(
    Pot,
    rmin=0.0,
    rmax=1.5,
    nrs=21,
    zmin=-0.5,
    zmax=0.5,
    nzs=21,
    phi=None,
    xy=False,
    xrange=None,
    yrange=None,
    t=0.0,
    effective=False,
    Lz=None,
    ncontours=21,
    savefilename=None,
    aspect=None,
    justcontours=False,
    levels=None,
    cntrcolors=None,
):
    """
    Plot a set of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential(s) to plot.
    rmin : float, optional
        Minimum R (can be Quantity). Default is 0.0.
    rmax : float, optional
        Maximum R (can be Quantity). Default is 1.5.
    nrs : int, optional
        Grid in R. Default is 21.
    zmin : float, optional
        Minimum z (can be Quantity). Default is -0.5.
    zmax : float, optional
        Maximum z (can be Quantity). Default is 0.5.
    nzs : int, optional
        Grid in z. Default is 21.
    phi : float, optional
        Azimuth to use for non-axisymmetric potentials. Default is None.
    t : float, optional
        Time to use to evaluate potential. Default is 0.0.
    xy : bool, optional
        If True, plot the potential in X-Y. Default is False.
    effective : bool, optional
        If True, plot the effective potential Phi + Lz^2/2/R^2. Default is False.
    Lz : float, optional
        Angular momentum to use for the effective potential when effective=True. Default is None.
    xrange : list, optional
        Minimum and maximum R values to plot. Default is None.
    yrange : list, optional
        Minimum and maximum z values to plot. Default is None.
    justcontours : bool, optional
        If True, just plot contours. Default is False.
    levels : array-like, optional
        Contours to plot. Default is None.
    ncontours : int, optional
        Number of contours when levels is None. Default is 21.
    cntrcolors : str or array-like, optional
        Colors of the contours (single color or array with length ncontours). Default is None.
    savefilename : str, optional
        Save to or restore from this savefile (pickle). Default is None.
    aspect : float, optional
        Aspect ratio of the plot. Default is None.

    Returns
    -------
    galpy.util.plot.dens2d return value

    Notes
    -----
    - 2010-07-09 - Written by Bovy (NYU).

    See Also
    --------
    galpy.util.plot.dens2d

    """
    if effective and xy:
        raise RuntimeError("xy and effective cannot be True at the same time")
    Pot = flatten(Pot)
    rmin = conversion.parse_length(rmin, **get_physical(Pot))
    rmax = conversion.parse_length(rmax, **get_physical(Pot))
    zmin = conversion.parse_length(zmin, **get_physical(Pot))
    zmax = conversion.parse_length(zmax, **get_physical(Pot))
    Lz = conversion.parse_angmom(Lz, **get_physical(Pot))
    if xrange is None:
        xrange = [rmin, rmax]
    if yrange is None:
        yrange = [zmin, zmax]
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        potRz = pickle.load(savefile)
        Rs = pickle.load(savefile)
        zs = pickle.load(savefile)
        savefile.close()
    else:
        if effective and Lz is None:
            raise RuntimeError("When effective=True, you need to specify Lz=")
        Rs = numpy.linspace(xrange[0], xrange[1], nrs)
        zs = numpy.linspace(yrange[0], yrange[1], nzs)
        potRz = numpy.zeros((nrs, nzs))
        for ii in range(nrs):
            for jj in range(nzs):
                if xy:
                    R, phi, z = coords.rect_to_cyl(Rs[ii], zs[jj], 0.0)
                else:
                    R, z = Rs[ii], zs[jj]
                potRz[ii, jj] = evaluatePotentials(
                    Pot, numpy.fabs(R), z, phi=phi, t=t, use_physical=False
                )
            if effective:
                potRz[ii, :] += 0.5 * Lz**2 / Rs[ii] ** 2.0
        # Don't plot outside of the desired range
        potRz[Rs < rmin, :] = numpy.nan
        potRz[Rs > rmax, :] = numpy.nan
        potRz[:, zs < zmin] = numpy.nan
        potRz[:, zs > zmax] = numpy.nan
        # Infinity is bad for plotting
        potRz[~numpy.isfinite(potRz)] = numpy.nan
        if not savefilename == None:
            print("Writing savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(potRz, savefile)
            pickle.dump(Rs, savefile)
            pickle.dump(zs, savefile)
            savefile.close()
    if aspect is None:
        aspect = 0.75 * (rmax - rmin) / (zmax - zmin)
    if xy:
        xlabel = r"$x/R_0$"
        ylabel = r"$y/R_0$"
    else:
        xlabel = r"$R/R_0$"
        ylabel = r"$z/R_0$"
    if levels is None:
        levels = numpy.linspace(numpy.nanmin(potRz), numpy.nanmax(potRz), ncontours)
    if cntrcolors is None:
        cntrcolors = "k"
    return plot.dens2d(
        potRz.T,
        origin="lower",
        cmap="gist_gray",
        contours=True,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect=aspect,
        xrange=xrange,
        yrange=yrange,
        cntrls="-",
        justcontours=justcontours,
        levels=levels,
        cntrcolors=cntrcolors,
    )


@potential_positional_arg
def plotDensities(
    Pot,
    rmin=0.0,
    rmax=1.5,
    nrs=21,
    zmin=-0.5,
    zmax=0.5,
    nzs=21,
    phi=None,
    xy=False,
    t=0.0,
    ncontours=21,
    savefilename=None,
    aspect=None,
    log=False,
    justcontours=False,
    **kwargs,
):
    """
    Plot the density of a set of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential(s) to evaluate.
    rmin : float, optional
        Minimum R (can be Quantity). Default is 0.0.
    rmax : float, optional
        Maximum R (can be Quantity). Default is 1.5.
    nrs : int, optional
        Grid in R. Default is 21.
    zmin : float, optional
        Minimum z (can be Quantity). Default is -0.5.
    zmax : float, optional
        Maximum z (can be Quantity). Default is 0.5.
    nzs : int, optional
        Grid in z. Default is 21.
    phi : float, optional
        Azimuth to use for non-axisymmetric potentials. Default is None.
    t : float, optional
        Time to use to evaluate potential. Default is 0.0.
    xy : bool, optional
        If True, plot the density in X-Y. Default is False.
    ncontours : int, optional
        Number of contours. Default is 21.
    justcontours : bool, optional
        If True, just plot contours. Default is False.
    savefilename : str, optional
        Save to or restore from this savefile (pickle). Default is None.
    log : bool, optional
        If True, plot the log density. Default is False.
    aspect : float, optional
        Aspect ratio of the plot. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to plot.dens2d.

    Returns
    -------
    galpy.util.plot.dens2d return value

    Notes
    -----
    - 2013-07-05 - Written - Bovy (IAS)
    - 2023-04-24 - Allow plotting in physical coordinates - Bovy (UofT)

    See Also
    --------
    galpy.util.plot.dens2d
    """
    Pot = flatten(Pot)
    physical_kwargs = conversion.extract_physical_kwargs(kwargs)
    use_physical, ro, vo = conversion.physical_output(Pot, physical_kwargs, "density")
    if ro is None:
        ro = get_physical(Pot)["ro"]
    physical_kwargs["quantity"] = False  # make sure to not use quantity output
    rmin = conversion.parse_length(rmin, ro=ro)
    rmax = conversion.parse_length(rmax, ro=ro)
    zmin = conversion.parse_length(zmin, ro=ro)
    zmax = conversion.parse_length(zmax, ro=ro)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        potRz = pickle.load(savefile)
        Rs = pickle.load(savefile)
        zs = pickle.load(savefile)
        savefile.close()
    else:
        Rs = numpy.linspace(rmin, rmax, nrs)
        zs = numpy.linspace(zmin, zmax, nzs)
        potRz = numpy.zeros((nrs, nzs))
        for ii in range(nrs):
            for jj in range(nzs):
                if xy:
                    R, phi, z = coords.rect_to_cyl(Rs[ii], zs[jj], 0.0)
                else:
                    R, z = Rs[ii], zs[jj]
                potRz[ii, jj] = evaluateDensities(
                    Pot, numpy.fabs(R), z, phi=phi, t=t, **physical_kwargs
                )
        if not savefilename == None:
            print("Writing savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(potRz, savefile)
            pickle.dump(Rs, savefile)
            pickle.dump(zs, savefile)
            savefile.close()
    if aspect is None:
        aspect = 0.75 * (rmax - rmin) / (zmax - zmin)
    if log:
        potRz = numpy.log(potRz)
    if xy:
        xlabel = r"$x"
        ylabel = r"$y"
    else:
        xlabel = r"$R"
        ylabel = r"$z"
    if use_physical:
        xlabel += r"\,(\mathrm{kpc})$"
        ylabel += r"\,(\mathrm{kpc})$"
    else:
        ro = 1.0
        xlabel += "/R_0$"
        ylabel += "/R_0$"
    kwargs["cmap"] = kwargs.get("cmap", "gist_yarg")
    return plot.dens2d(
        potRz.T,
        origin="lower",
        contours=True,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect=aspect,
        xrange=[rmin * ro, rmax * ro],
        yrange=[zmin * ro, zmax * ro],
        cntrls=kwargs.pop("cntrls", "-"),
        justcontours=justcontours,
        levels=numpy.linspace(numpy.nanmin(potRz), numpy.nanmax(potRz), ncontours),
        **kwargs,
    )


@potential_positional_arg
def plotSurfaceDensities(
    Pot,
    xmin=-1.5,
    xmax=1.5,
    nxs=21,
    ymin=-1.5,
    ymax=1.5,
    nys=21,
    z=numpy.inf,
    t=0.0,
    ncontours=21,
    savefilename=None,
    aspect=None,
    log=False,
    justcontours=False,
    **kwargs,
):
    """
    Plot the surface density of a set of potentials.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential(s) for which to plot the surface density.
    xmin : float or Quantity
        Minimum x value. Default is -1.5.
    xmax : float or Quantity
        Maximum x value. Default is 1.5.
    nxs : int
        Number of grid points in x.
    ymin : float or Quantity
        Minimum y value. Default is -1.5.
    ymax : float or Quantity
        Maximum y value. Default is 1.5.
    nys : int
        Number of grid points in y.
    z : float or Quantity, optional
        Height between which to integrate the density (from -z to z). Default is numpy.inf.
    t : float, optional
        Time to use to evaluate potential. Default is 0.0.
    ncontours : int, optional
        Number of contours. Default is 21.
    justcontours : bool, optional
        If True, just plot contours. Default is False.
    aspect : float, optional
        Aspect ratio of the plot. Default is None.
    savefilename : str, optional
        Save to or restore from this savefile (pickle). Default is None.
    log : bool, optional
        If True, plot the log density. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to plot.dens2d.

    Returns
    -------
    matplotlib plot
        Plot to output device.

    Notes
    -----
    - 2020-08-19 - Written - Bovy (UofT)

    """
    Pot = flatten(Pot)
    physical_kwargs = conversion.extract_physical_kwargs(kwargs)
    use_physical, ro, vo = conversion.physical_output(
        Pot, physical_kwargs, "surfacedensity"
    )
    if ro is None:
        ro = get_physical(Pot)["ro"]
    physical_kwargs["quantity"] = False  # make sure to not use quantity output
    xmin = conversion.parse_length(xmin, ro=ro)
    xmax = conversion.parse_length(xmax, ro=ro)
    ymin = conversion.parse_length(ymin, ro=ro)
    ymax = conversion.parse_length(ymax, ro=ro)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        surfxy = pickle.load(savefile)
        xs = pickle.load(savefile)
        ys = pickle.load(savefile)
        savefile.close()
    else:
        xs = numpy.linspace(xmin, xmax, nxs)
        ys = numpy.linspace(ymin, ymax, nys)
        surfxy = numpy.zeros((nxs, nys))
        for ii in range(nxs):
            for jj in range(nys):
                R, phi, _ = coords.rect_to_cyl(xs[ii], ys[jj], 0.0)
                surfxy[ii, jj] = evaluateSurfaceDensities(
                    Pot, numpy.fabs(R), z, phi=phi, t=t, **physical_kwargs
                )
        if not savefilename == None:
            print("Writing savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(surfxy, savefile)
            pickle.dump(xs, savefile)
            pickle.dump(ys, savefile)
            savefile.close()
    if aspect is None:
        aspect = 1.0
    if log:
        surfxy = numpy.log(surfxy)
    xlabel = r"$x"
    ylabel = r"$y"
    if use_physical:
        xlabel += r"\,(\mathrm{kpc})$"
        ylabel += r"\,(\mathrm{kpc})$"
    else:
        ro = 1.0
        xlabel += "/R_0$"
        ylabel += "/R_0$"
    kwargs["cmap"] = kwargs.get("cmap", "gist_yarg")
    return plot.dens2d(
        surfxy.T,
        origin="lower",
        contours=True,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect=aspect,
        xrange=[xmin * ro, xmax * ro],
        yrange=[ymin * ro, ymax * ro],
        cntrls=kwargs.pop("cntrls", "-"),
        justcontours=justcontours,
        levels=numpy.linspace(numpy.nanmin(surfxy), numpy.nanmax(surfxy), ncontours),
        **kwargs,
    )


@potential_positional_arg
@potential_physical_input
@physical_conversion("frequency", pop=True)
def epifreq(Pot, R, t=0.0):
    """
    Calculate the epicycle frequency at R in the potential Pot.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    R : float or Quantity
        Galactocentric radius.
    t : float or Quantity, optional
        Time (default: 0).

    Returns
    -------
    float or Quantity
        Epicycle frequency.

    Notes
    -----
    - 2012-07-25 - Written - Bovy (IAS)

    """
    from .planarPotential import planarPotential

    if isinstance(Pot, (Potential, planarPotential)):
        return Pot.epifreq(R, t=t, use_physical=False)
    from ..potential import (
        PotentialError,
        evaluateplanarR2derivs,
        evaluateplanarRforces,
    )

    try:
        return numpy.sqrt(
            evaluateplanarR2derivs(Pot, R, t=t, use_physical=False)
            - 3.0 / R * evaluateplanarRforces(Pot, R, t=t, use_physical=False)
        )
    except PotentialError:
        from ..potential import RZToplanarPotential

        Pot = RZToplanarPotential(Pot)
        return numpy.sqrt(
            evaluateplanarR2derivs(Pot, R, t=t, use_physical=False)
            - 3.0 / R * evaluateplanarRforces(Pot, R, t=t, use_physical=False)
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("frequency", pop=True)
def verticalfreq(Pot, R, t=0.0):
    """
    Calculate the vertical frequency at R in the potential Pot.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    R : float or Quantity
        Galactocentric radius.
    t : float or Quantity, optional
        Time (default: 0).

    Returns
    -------
    float or Quantity
        Vertical frequency.

    Notes
    -----
    - 2012-07-25 - Written - Bovy (IAS@MPIA)

    """
    from .planarPotential import planarPotential

    if isinstance(Pot, (Potential, planarPotential)):
        return Pot.verticalfreq(R, t=t, use_physical=False)
    return numpy.sqrt(evaluatez2derivs(Pot, R, 0.0, t=t, use_physical=False))


@potential_positional_arg
@potential_physical_input
@physical_conversion("dimensionless", pop=True)
def flattening(Pot, R, z, t=0.0):
    """
    Calculate the potential flattening, defined as sqrt(fabs(z/R F_R/F_z))

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    R : float or Quantity
        Galactocentric radius.
    z : float or Quantity
        Height.
    t : float or Quantity, optional
        Time (default: 0).

    Returns
    -------
    float or Quantity
        Flattening.

    Notes
    -----
    - 2012-09-13 - Written - Bovy (IAS)

    """
    return numpy.sqrt(
        numpy.fabs(
            z
            / R
            * evaluateRforces(Pot, R, z, t=t, use_physical=False)
            / evaluatezforces(Pot, R, z, t=t, use_physical=False)
        )
    )


@potential_positional_arg
@physical_conversion("velocity", pop=True)
def vterm(Pot, l, t=0.0, deg=True):
    """
    Calculate the terminal velocity at l in this potential.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    l : float or Quantity
        Galactic longitude [deg/rad; can be Quantity).
    t : float or Quantity, optional
        Time (default: 0).
    deg : bool, optional
        If True (default), l in deg.

    Returns
    -------
    float or Quantity
        Terminal velocity.

    Notes
    -----
    - 2013-05-31 - Written - Bovy (IAS)

    """
    Pot = flatten(Pot)
    if _APY_LOADED and isinstance(l, units.Quantity):
        l = conversion.parse_angle(l)
        deg = False
    if deg:
        sinl = numpy.sin(l / 180.0 * numpy.pi)
    else:
        sinl = numpy.sin(l)
    return sinl * (
        omegac(Pot, sinl, t=t, use_physical=False)
        - omegac(Pot, 1.0, t=t, use_physical=False)
    )


@potential_positional_arg
@physical_conversion("position", pop=True)
def rl(Pot, lz, t=0.0):
    """
    Calculate the radius of a circular orbit of Lz.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    lz : float or Quantity
        Angular momentum (can be Quantity).
    t : float or Quantity, optional
        Time (default: 0).

    Returns
    -------
    float or Quantity
        Radius.

    Notes
    -----
    - 2012-07-30 - Written - Bovy (IAS@MPIA)

    - An efficient way to call this function on many objects is provided as the Orbit method rguiding.

    See Also
    --------
    Orbit.rguiding

    """
    Pot = flatten(Pot)
    lz = conversion.parse_angmom(lz, **conversion.get_physical(Pot))
    # Find interval
    rstart = _rlFindStart(numpy.fabs(lz), numpy.fabs(lz), Pot, t=t)  # assumes vo=1.
    try:
        return optimize.brentq(
            _rlfunc,
            10.0**-5.0,
            rstart,
            args=(numpy.fabs(lz), Pot, t),
            maxiter=200,
            disp=False,
        )
    except ValueError:  # Probably lz small and starting lz to great
        rlower = _rlFindStart(10.0**-5.0, numpy.fabs(lz), Pot, t=t, lower=True)
        return optimize.brentq(_rlfunc, rlower, rstart, args=(numpy.fabs(lz), Pot, t))


def _rlfunc(rl, lz, pot, t=0.0):
    """Function that gives rvc-lz"""
    thisvcirc = vcirc(pot, rl, t=t, use_physical=False)
    return rl * thisvcirc - lz


def _rlFindStart(rl, lz, pot, t=0.0, lower=False):
    """find a starting interval for rl"""
    rtry = 2.0 * rl
    while (2.0 * lower - 1.0) * _rlfunc(rtry, lz, pot, t=t) > 0.0:
        if lower:
            rtry /= 2.0
        else:
            rtry *= 2.0
    return rtry


@potential_positional_arg
@physical_conversion("position", pop=True)
def rE(Pot, E, t=0.0):
    """
    Calculate the radius of a circular orbit with energy E.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    E : float or Quantity
        Energy.
    t : float, optional
        Time (default is 0.0).

    Returns
    -------
    radius : float
        Radius.

    Notes
    -----
    - 2022-04-06 - Written - Bovy (UofT)

    - An efficient way to call this function on many objects is provided as the Orbit method rE.

    See Also
    --------
    Orbit.rE

    """
    Pot = flatten(Pot)
    E = conversion.parse_energy(E, **conversion.get_physical(Pot))
    # Find interval
    rstart = _rEFindStart(1.0, E, Pot, t=t)
    try:
        return optimize.brentq(
            _rEfunc, 10.0**-5.0, rstart, args=(E, Pot, t), maxiter=200, disp=False
        )
    except ValueError:  # Probably E small and starting rE to great
        rlower = _rEFindStart(10.0**-5.0, E, Pot, t=t, lower=True)
        return optimize.brentq(_rEfunc, rlower, rstart, args=(E, Pot, t))


def _rEfunc(rE, E, pot, t=0.0):
    """Function that gives vc^2/2+Pot(rc)-E"""
    thisvcirc = vcirc(pot, rE, t=t, use_physical=False)
    return thisvcirc**2.0 / 2.0 + _evaluatePotentials(pot, rE, 0.0, t=t) - E


def _rEFindStart(rE, E, pot, t=0.0, lower=False):
    """find a starting interval for rE"""
    rtry = 2.0 * rE
    while (2.0 * lower - 1.0) * _rEfunc(rtry, E, pot, t=t) > 0.0:
        if lower:
            rtry /= 2.0
        else:
            rtry *= 2.0
    return rtry


@potential_positional_arg
@physical_conversion("action", pop=True)
def LcE(Pot, E, t=0.0):
    """
    Calculate the angular momentum of a circular orbit with energy E.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list thereof.
    E : float or Quantity
        Energy.
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        Angular momentum of circular orbit with energy E

    Notes
    -----
    - 2022-04-06 - Written - Bovy (UofT)

    """
    thisrE = rE(Pot, E, t=t, use_physical=False)
    return thisrE * vcirc(Pot, thisrE, use_physical=False)


@potential_positional_arg
@physical_conversion("position", pop=True)
def lindbladR(Pot, OmegaP, m=2, t=0.0, **kwargs):
    """
    Calculate the radius of a Lindblad resonance.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances.
    OmegaP : float or Quantity
        Pattern speed.
    m : int or str, optional
        Order of the resonance (as in m(O-Op)=kappa (negative m for outer)).
        Use m='corotation' for corotation (default: 2).
    t : float or Quantity, optional
        Time (default: 0.0).
    **kwargs
        Additional arguments to be passed to scipy.optimize.brentq.

    Returns
    -------
    float or Quantity or None
        Radius of Lindblad resonance, None if there is no resonance.

    Notes
    -----
    - 2011-10-09 - Written - Bovy (IAS)

    """
    Pot = flatten(Pot)
    OmegaP = conversion.parse_frequency(OmegaP, **conversion.get_physical(Pot))
    if isinstance(m, str):
        if "corot" in m.lower():
            corotation = True
        else:
            raise OSError(
                "'m' input not recognized, should be an integer or 'corotation'"
            )
    else:
        corotation = False
    if corotation:
        try:
            out = optimize.brentq(
                _corotationR_eq, 0.0000001, 1000.0, args=(Pot, OmegaP, t), **kwargs
            )
        except ValueError:
            try:
                # Sometimes 0.0000001 is numerically too small to start...
                out = optimize.brentq(
                    _corotationR_eq, 0.01, 1000.0, args=(Pot, OmegaP, t), **kwargs
                )
            except ValueError:
                return None
        except RuntimeError:  # pragma: no cover
            raise
        return out
    else:
        try:
            out = optimize.brentq(
                _lindbladR_eq, 0.0000001, 1000.0, args=(Pot, OmegaP, m, t), **kwargs
            )
        except ValueError:
            return None
        except RuntimeError:  # pragma: no cover
            raise
        return out


def _corotationR_eq(R, Pot, OmegaP, t=0.0):
    return omegac(Pot, R, t=t, use_physical=False) - OmegaP


def _lindbladR_eq(R, Pot, OmegaP, m, t=0.0):
    return m * (omegac(Pot, R, t=t, use_physical=False) - OmegaP) - epifreq(
        Pot, R, t=t, use_physical=False
    )


@potential_positional_arg
@potential_physical_input
@physical_conversion("frequency", pop=True)
def omegac(Pot, R, t=0.0):
    """
    Calculate the circular angular speed velocity at R in potential Pot.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances.
    R : float or Quantity
        Galactocentric radius.
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float or Quantity
        Circular angular speed.

    Notes
    -----
    - 2011-10-09 - Written - Bovy (IAS)

    """
    from ..potential import evaluateplanarRforces

    try:
        return numpy.sqrt(-evaluateplanarRforces(Pot, R, t=t, use_physical=False) / R)
    except PotentialError:
        from ..potential import RZToplanarPotential

        Pot = RZToplanarPotential(Pot)
        return numpy.sqrt(-evaluateplanarRforces(Pot, R, t=t, use_physical=False) / R)


def nemo_accname(Pot):
    """
    Return the accname potential name for use of this potential or list of potentials with NEMO.

    Parameters
    ----------
    Pot : Potential instance or list of such instances

    Returns
    -------
    str
        Acceleration name in the correct format to give to accname=

    Notes
    -----
    - 2014-12-18 - Written - Bovy (IAS)

    """
    Pot = flatten(Pot)
    if isinstance(Pot, list):
        out = ""
        for ii, pot in enumerate(Pot):
            if ii > 0:
                out += "+"
            out += pot.nemo_accname()
        return out
    elif isinstance(Pot, Potential):
        return Pot.nemo_accname()
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'nemo_accname' is neither a Potential-instance or a list of such instances"
        )


def nemo_accpars(Pot, vo, ro):
    """
    Return the accpars potential parameters for use of this potential or list of potentials with NEMO.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    str
        Accpars string in the correct format to give to accpars.

    Notes
    -----
    - 2014-12-18 - Written - Bovy (IAS)

    """
    Pot = flatten(Pot)
    if isinstance(Pot, list):
        out = ""
        for ii, pot in enumerate(Pot):
            if ii > 0:
                out += "#"
            out += pot.nemo_accpars(vo, ro)
        return out
    elif isinstance(Pot, Potential):
        return Pot.nemo_accpars(vo, ro)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'nemo_accpars' is neither a Potential-instance or a list of such instances"
        )


def to_amuse(
    Pot, t=0.0, tgalpy=0.0, reverse=False, ro=None, vo=None
):  # pragma: no cover
    """
    Return an AMUSE representation of a galpy Potential or list of Potentials

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential(s) to convert to an AMUSE representation.
    t : float, optional
        Initial time in AMUSE (can be in internal galpy units or AMUSE units), by default 0.0.
    tgalpy : float, optional
        Initial time in galpy (can be in internal galpy units or AMUSE units); because AMUSE initial times have to be positive, this is useful to set if the initial time in galpy is negative, by default 0.0.
    reverse : bool, optional
        Set whether the galpy potential evolves forwards or backwards in time (default: False); because AMUSE can only integrate forward in time, this is useful to integrate backward in time in AMUSE, by default False.
    ro : float, optional
        Length unit in kpc, by default None.
    vo : float, optional
        Velocity unit in km/s, by default None.

    Returns
    -------
    AMUSE representation of Pot.

    Notes
    -----
    - 2019-08-04 - Written - Bovy (UofT)
    - 2019-08-12 - Implemented actual function - Webb (UofT)

    """
    try:
        from . import amuse
    except ImportError:
        raise ImportError(
            "To obtain an AMUSE representation of a galpy potential, you need to have AMUSE installed"
        )
    Pot = flatten(Pot)
    if ro is None or vo is None:
        physical_dict = get_physical(Pot)
        if ro is None:
            ro = physical_dict.get("ro")
        if vo is None:
            vo = physical_dict.get("vo")
    return amuse.galpy_profile(Pot, t=t, tgalpy=tgalpy, ro=ro, vo=vo)


def turn_physical_off(Pot):
    """
    Turn off automatic returning of outputs in physical units.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential(s) to turn off automatic returning of outputs in physical units.

    Returns
    -------
    None

    Notes
    -----
    - 2016-01-30 - Written - Bovy (UofT)

    """
    if isinstance(Pot, list):
        for pot in Pot:
            turn_physical_off(pot)
    else:
        Pot.turn_physical_off()
    return None


def turn_physical_on(Pot, ro=None, vo=None):
    """
    Turn on automatic returning of outputs in physical units.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential(s) to turn on automatic returning of outputs in physical units.
    ro : float or Quantity, optional
        Reference distance (kpc). Default is None.
    vo : float or Quantity, optional
        Reference velocity (km/s). Default is None.

    Returns
    -------
    None

    Notes
    -----
    - 2016-01-30 - Written - Bovy (UofT)

    """
    if isinstance(Pot, list):
        for pot in Pot:
            turn_physical_on(pot, ro=ro, vo=vo)
    else:
        Pot.turn_physical_on(ro=ro, vo=vo)
    return None


def _flatten_list(L):
    for item in L:
        try:
            yield from _flatten_list(item)
        except TypeError:
            yield item


def flatten(Pot):
    """
    Flatten a possibly nested list of Potential instances into a flat list.

    Parameters
    ----------
    Pot : list or Potential instance
        List (possibly nested) of Potential instances.

    Returns
    -------
    list
        Flattened list of Potential instances.

    Notes
    -----
    - 2018-03-14 - Written - Bovy (UofT).

    """
    if isinstance(Pot, Potential):
        return Pot
    elif isinstance(Pot, list):
        return list(_flatten_list(Pot))
    else:
        return Pot


def _check_c(Pot, dxdv=False, dens=False):
    """
    Check whether a potential or list thereof has a C implementation.

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances to check.
    dxdv : bool, optional
        If True, check whether the potential has dxdv implementation.
    dens : bool, optional
        If True, check whether the potential has its density implemented in C.

    Returns
    -------
    bool
        True if a C implementation exists, False otherwise.

    Notes
    -----
    - 2014-02-17 - Written - Bovy (IAS)
    - 2017-07-01 - Generalized to dxdv, added general support for WrapperPotentials, and added support for planarPotentials.

    """
    Pot = flatten(Pot)
    from ..potential import linearPotential, planarForce

    if dxdv:
        hasC_attr = "hasC_dxdv"
    elif dens:
        hasC_attr = "hasC_dens"
    else:
        hasC_attr = "hasC"
    from .WrapperPotential import parentWrapperPotential

    if isinstance(Pot, list):
        return numpy.all(
            numpy.array([_check_c(p, dxdv=dxdv, dens=dens) for p in Pot], dtype="bool")
        )
    elif isinstance(Pot, parentWrapperPotential):
        return bool(Pot.__dict__[hasC_attr] * _check_c(Pot._pot))
    elif (
        isinstance(Pot, Force)
        or isinstance(Pot, planarForce)
        or isinstance(Pot, linearPotential)
    ):
        return Pot.__dict__[hasC_attr]


def _dim(Pot):
    """
    Determine the dimensionality of this potential

    Parameters
    ----------
    Pot : Potential instance or list of such instances

    Returns
    -------
    int
        Minimum of the dimensionality of all potentials if list; otherwise Pot.dim

    Notes
    -----
    - 2016-04-19 - Written - Bovy (UofT)

    """
    from ..potential import linearPotential, planarPotential

    if isinstance(Pot, list):
        return numpy.amin(numpy.array([_dim(p) for p in Pot], dtype="int"))
    elif isinstance(
        Pot, (Potential, planarPotential, linearPotential, DissipativeForce)
    ):
        return Pot.dim


def _isNonAxi(Pot):
    """
    Determine whether this potential is non-axisymmetric

    Parameters
    ----------
    Pot : Potential instance or list of such instances

    Returns
    -------
    bool
        True or False depending on whether the potential is non-axisymmetric (note that some potentials might return True, even though for some parameter values they are axisymmetric)

    Notes
    -----
    - 2016-06-16 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    if isList:
        isAxis = [not _isNonAxi(p) for p in Pot]
        nonAxi = not numpy.prod(numpy.array(isAxis))
    else:
        nonAxi = Pot.isNonAxi
    return nonAxi


def kms_to_kpcGyrDecorator(func):
    """Decorator to convert velocities from km/s to kpc/Gyr"""

    @wraps(func)
    def kms_to_kpcGyr_wrapper(*args, **kwargs):
        return func(args[0], velocity_in_kpcGyr(args[1], 1.0), args[2], **kwargs)

    return kms_to_kpcGyr_wrapper


@potential_positional_arg
@potential_physical_input
@physical_conversion("position", pop=True)
def rtide(Pot, R, z, phi=0.0, t=0.0, M=None):
    """
    Calculate the tidal radius for object of mass M assuming a circular orbit as

        .. math::

           r_t^3 = \\frac{GM_s}{\\Omega^2-\\mathrm{d}^2\\Phi/\\mathrm{d}r^2}

        where :math:`M_s` is the cluster mass, :math:`\\Omega` is the circular frequency, and :math:`\\Phi` is the gravitational potential. For non-spherical potentials, we evaluate :math:`\\Omega^2 = (1/r)(\\mathrm{d}\\Phi/\\mathrm{d}r)` and evaluate the derivatives at the given position of the cluster.

    Parameters
    ----------
    R : float or Quantity
        Galactocentric radius
    z : float or Quantity
        height
    phi : float or Quantity, optional
        azimuth (default: 0.0)
    t : float or Quantity, optional
        time (default: 0.0)
    M : float or Quantity, optional
        Mass of object (default: None)

    Returns
    -------
    float or Quantity
        Tidal Radius

    Notes
    -----
    - 2018-03-21 - Written - Webb (UofT)

    """
    Pot = flatten(Pot)
    if M is None:
        # Make sure an object mass is given
        raise PotentialError(
            "Mass parameter M= needs to be set to compute tidal radius"
        )
    r = numpy.sqrt(R**2.0 + z**2.0)
    omegac2 = -evaluaterforces(Pot, R, z, phi=phi, t=t, use_physical=False) / r
    d2phidr2 = evaluater2derivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    return (M / (omegac2 - d2phidr2)) ** (1.0 / 3.0)


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def ttensor(Pot, R, z, phi=0.0, t=0.0, eigenval=False):
    """
    Calculate the tidal tensor Tij=-d(Psi)(dxidxj)

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances
    R : float or Quantity
        Galactocentric radius
    z : float or Quantity
        height
    phi : float or Quantity, optional
        azimuth (default: 0.0)
    t : float or Quantity, optional
        time (default: 0.0)
    eigenval : bool, optional
        return eigenvalues if true (default: False)

    Returns
    -------
    array, shape (3,3) or (3,)
        Tidal tensor or eigenvalues of the tidal tensor

    Notes
    -----
    - 2018-03-21 - Written - Webb (UofT)

    """
    Pot = flatten(Pot)
    if _isNonAxi(Pot):
        raise PotentialError(
            "Tidal tensor calculation is currently only implemented for axisymmetric potentials"
        )
    # Evaluate forces, angles and derivatives
    Rderiv = -evaluateRforces(Pot, R, z, phi=phi, t=t, use_physical=False)
    phideriv = -evaluatephitorques(Pot, R, z, phi=phi, t=t, use_physical=False)
    R2deriv = evaluateR2derivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    z2deriv = evaluatez2derivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    phi2deriv = evaluatephi2derivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    Rzderiv = evaluateRzderivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    Rphideriv = evaluateRphiderivs(Pot, R, z, phi=phi, t=t, use_physical=False)
    # Temporarily set zphideriv to zero until zphideriv is added to Class
    zphideriv = 0.0
    cosphi = numpy.cos(phi)
    sinphi = numpy.sin(phi)
    cos2phi = cosphi**2.0
    sin2phi = sinphi**2.0
    R2 = R**2.0
    R3 = R**3.0
    # Tidal tensor
    txx = (
        R2deriv * cos2phi
        - Rphideriv * 2.0 * cosphi * sinphi / R
        + Rderiv * sin2phi / R
        + phi2deriv * sin2phi / R2
        + phideriv * 2.0 * cosphi * sinphi / R2
    )
    tyx = (
        R2deriv * sinphi * cosphi
        + Rphideriv * (cos2phi - sin2phi) / R
        - Rderiv * sinphi * cosphi / R
        - phi2deriv * sinphi * cosphi / R2
        + phideriv * (sin2phi - cos2phi) / R2
    )
    tzx = Rzderiv * cosphi - zphideriv * sinphi / R
    tyy = (
        R2deriv * sin2phi
        + Rphideriv * 2.0 * cosphi * sinphi / R
        + Rderiv * cos2phi / R
        + phi2deriv * cos2phi / R2
        - phideriv * 2.0 * sinphi * cosphi / R2
    )
    txy = tyx
    tzy = Rzderiv * sinphi + zphideriv * cosphi / R
    txz = tzx
    tyz = tzy
    tzz = z2deriv
    tij = -numpy.array([[txx, txy, txz], [tyx, tyy, tyz], [tzx, tzy, tzz]])
    if eigenval:
        return numpy.linalg.eigvals(tij)
    else:
        return tij


@potential_positional_arg
@physical_conversion("position", pop=True)
def zvc(Pot, R, E, Lz, phi=0.0, t=0.0):
    """
    Calculate the zero-velocity curve: z such that Phi(R,z) + Lz/[2R^2] = E (assumes that F_z(R,z) = negative at positive z such that there is a single solution)

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances.
    R : float or Quantity
        Galactocentric radius.
    E : float or Quantity
        Energy.
    Lz : float or Quantity
        Angular momentum.
    phi : float or Quantity, optional
        Azimuth (default: 0.0).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float
        z such that Phi(R,z) + Lz/[2R^2] = E.

    Notes
    -----
    - 2020-08-20 - Written - Bovy (UofT)

    """
    Pot = flatten(Pot)
    R = conversion.parse_length(R, **get_physical(Pot))
    E = conversion.parse_energy(E, **get_physical(Pot))
    Lz = conversion.parse_angmom(Lz, **get_physical(Pot))
    Lz2over2R2 = Lz**2.0 / 2.0 / R**2.0
    # Check z=0 and whether a solution exists
    if (
        numpy.fabs(_evaluatePotentials(Pot, R, 0.0, phi=phi, t=t) + Lz2over2R2 - E)
        < 1e-8
    ):
        return 0.0
    elif _evaluatePotentials(Pot, R, 0.0, phi=phi, t=t) + Lz2over2R2 > E:
        return numpy.nan  # s.t. this does not get plotted
    # Find starting value
    zstart = 1.0
    zmax = 1000.0
    while (
        E - _evaluatePotentials(Pot, R, zstart, phi=phi, t=t) - Lz2over2R2 > 0.0
        and zstart < zmax
    ):
        zstart *= 2.0
    try:
        out = optimize.brentq(
            lambda z: _evaluatePotentials(Pot, R, z, phi=phi, t=t) + Lz2over2R2 - E,
            0.0,
            zstart,
        )
    except ValueError:
        raise ValueError(
            "No solution for the zero-velocity curve found for this combination of parameters"
        )
    return out


@potential_positional_arg
@physical_conversion("position", pop=True)
def zvc_range(Pot, E, Lz, phi=0.0, t=0.0):
    """
    Calculate the minimum and maximum radius for which the zero-velocity curve exists for this energy and angular momentum (R such that Phi(R,0) + Lz/[2R^2] = E)

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Potential instance or list of such instances.
    E : float or Quantity
        Energy.
    Lz : float or Quantity
        Angular momentum.
    phi : float or Quantity, optional
        Azimuth (default: 0.0).
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    numpy.ndarray
        Solutions R such that Phi(R,0) + Lz/[2R^2] = E.

    Notes
    -----
    - 2020-08-20 - Written - Bovy (UofT)
    """
    Pot = flatten(Pot)
    E = conversion.parse_energy(E, **get_physical(Pot))
    Lz = conversion.parse_angmom(Lz, **get_physical(Pot))
    Lz2over2 = Lz**2.0 / 2.0
    # Check whether a solution exists
    RLz = rl(Pot, Lz, t=t, use_physical=False)
    Rstart = RLz
    if _evaluatePotentials(Pot, Rstart, 0.0, phi=phi, t=t) + Lz2over2 / Rstart**2.0 > E:
        return numpy.array([numpy.nan, numpy.nan])
    # Find starting value for Rmin
    Rstartmin = 1e-8
    while (
        _evaluatePotentials(Pot, Rstart, 0, phi=phi, t=t) + Lz2over2 / Rstart**2.0 < E
        and Rstart > Rstartmin
    ):
        Rstart /= 2.0
    Rmin = optimize.brentq(
        lambda R: _evaluatePotentials(Pot, R, 0, phi=phi, t=t) + Lz2over2 / R**2.0 - E,
        Rstart,
        RLz,
    )
    # Find starting value for Rmax
    Rstart = RLz
    Rstartmax = 1000.0
    while (
        _evaluatePotentials(Pot, Rstart, 0, phi=phi, t=t) + Lz2over2 / Rstart**2.0 < E
        and Rstart < Rstartmax
    ):
        Rstart *= 2.0
    Rmax = optimize.brentq(
        lambda R: _evaluatePotentials(Pot, R, 0, phi=phi, t=t) + Lz2over2 / R**2.0 - E,
        RLz,
        Rstart,
    )
    return numpy.array([Rmin, Rmax])


@potential_positional_arg
@physical_conversion("position", pop=True)
def rhalf(Pot, t=0.0, INF=numpy.inf):
    """
    Calculate the half-mass radius, the radius of the spherical shell that contains half the total mass.

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list of instances.
    t : float or Quantity, optional
        Time (default: 0.0).
    INF : numeric, optional
        Radius at which the total mass is calculated (internal units, just set this to something very large) (default: numpy.inf).

    Returns
    -------
    float
        Half-mass radius.

    Notes
    -----
    - 2021-03-18 - Written - Bovy (UofT)

    """
    Pot = flatten(Pot)
    tot_mass = mass(Pot, INF, t=t)
    # Find interval
    rhi = _rhalfFindStart(1.0, Pot, tot_mass, t=t)
    rlo = _rhalfFindStart(1.0, Pot, tot_mass, t=t, lower=True)
    return optimize.brentq(
        _rhalffunc, rlo, rhi, args=(Pot, tot_mass, t), maxiter=200, disp=False
    )


def _rhalffunc(rh, pot, tot_mass, t=0.0):
    return mass(pot, rh, t=t) / tot_mass - 0.5


def _rhalfFindStart(rh, pot, tot_mass, t=0.0, lower=False):
    """find a starting interval for rhalf"""
    rtry = 2.0 * rh
    while (2.0 * lower - 1.0) * _rhalffunc(rtry, pot, tot_mass, t=t) > 0.0:
        if lower:
            rtry /= 2.0
        else:
            rtry *= 2.0
    return rtry


@potential_positional_arg
@potential_physical_input
@physical_conversion("time", pop=True)
def tdyn(Pot, R, t=0.0):
    """
    Calculate the dynamical time from tdyn^2 = 3pi/[G<rho>].

    Parameters
    ----------
    Pot : Potential instance or list thereof
        Potential instance or list of instances.
    R : float or Quantity
        Galactocentric radius.
    t : float or Quantity, optional
        Time (default: 0.0).

    Returns
    -------
    float
        Dynamical time.

    Notes
    -----
    - 2021-03-18 - Written - Bovy (UofT)

    """
    return 2.0 * numpy.pi * R * numpy.sqrt(R / mass(Pot, R, use_physical=False))
