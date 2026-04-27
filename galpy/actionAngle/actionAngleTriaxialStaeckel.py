###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckelTriaxial
#
#             Use Sanders & Binney (2015; MNRAS 447/3/2479)'s Triaxial Staeckel Fudge approximation for
#             calculating the actions
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import copy
import warnings

import numpy
from scipy import integrate, optimize

from ..potential import (
    CompositePotential,
    DiskSCFPotential,
    MWPotential,
    SCFPotential,
    epifreq,
    evaluateR2derivs,
    evaluateRzderivs,
    evaluatez2derivs,
    omegac,
    verticalfreq,
)
from ..potential.Potential import (
    PotentialError,
    _check_c,
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    _isNonAxi,
)
from ..util import coords  # for prolate confocal transforms
from ..util import conversion, galpyWarning
from ..util.conversion import physical_conversion, potential_physical_input
#from . import actionAngleStaeckelTriaxial_c
from .actionAngle import UnboundError, actionAngle
#from .actionAngleStaeckelTriaxial_c import _ext_loaded as ext_loaded


class actionAngleStaeckelTriaxial(actionAngle):
    """Action-angle formalism for triaxial potentials using Sanders & Binney (2015)'s Triaxial Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckelTriaxial object.

        Parameters
        ----------
        pot : potential or a combined potential formed using addition (pot1+pot2+…) (3D)
            The potential or a combined potential formed using addition (pot1+pot2+…).
        c : bool, optional
            If True, always use C for calculations. Default is False.
        order : int, optional
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals. Default is 10.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2026-04-06 - Started - Weatherall.
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelTriaxial")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        
        ext_loaded = False
        if ext_loaded and (("c" in kwargs and kwargs["c"]) or not "c" in kwargs):
            self._c = _check_c(self._pot)
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
        else:
            self._c = False
        self._order = kwargs.get("order", 10)
        # Check the units
        self._check_consistent_units()
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz,phi:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2026-04-06 - Started - Weatherall
        """
        order = kwargs.get("order", self._order)
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        if len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
            phi = self._eval_phi
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
            phi = numpy.array([phi])
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            #or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            #Lz = R * vT
            return 0, 0, 0
            #jr, jz, err = actionAngleStaeckelTriaxialFudge_c.actionAngleStaeckelTriaxialFudge_c(
            #    self._pot, R, vR, vT, z, vz, phi, order=order
            #)
            #if err == 0:
            #    return (jr, Lz, jz)
            #else:  # pragma: no cover
            #    raise RuntimeError(
            #        "C-code for calculation actions failed; try with c=False"
            #    )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            kwargs.pop("c", None)
            if len(R) > 1:
                ojr = numpy.zeros(len(R))
                olz = numpy.zeros(len(R))
                ojz = numpy.zeros(len(R))
                for ii in range(len(R)):
                    targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii], phi[ii])
                    tkwargs = copy.copy(kwargs)

                    tjr, tlz, tjz = self(*targs, **tkwargs)
                    ojr[ii] = tjr[0]
                    ojz[ii] = tjz[0]
                    olz[ii] = tlz[0]
                return (ojr, olz, ojz)
            else:
                # Set up the actionAngleStaeckelTriaxialFudgeSingle object
                aASingle = actionAngleStaeckelTriaxialSingle(
                    R[0],
                    vR[0],
                    vT[0],
                    z[0],
                    vz[0],
                    phi[0],
                    pot=self._pot,
                )
                return (
                    numpy.atleast_1d(aASingle.JR(**copy.copy(kwargs))),
                    numpy.atleast_1d(aASingle.Jphi(**copy.copy(kwargs))),
                    numpy.atleast_1d(aASingle.Jz(**copy.copy(kwargs))),
                )

class actionAngleStaeckelTriaxialSingle(actionAngle):
    """Action-angle formalism for triaxial potentials using Sanders & Binney (2015)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckelTriaxialFudgeSingle object

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz,phi:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        pot: Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential to use
        
        Notes
        -----
        
        """
        self._parse_eval_args(*args, _noOrbUnitsCheck=True, **kwargs)
        self._R = self._eval_R
        self._vR = self._eval_vR
        self._vT = self._eval_vT
        self._z = self._eval_z
        self._vz = self._eval_vz
        self._phi = self._eval_phi
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelTriaxialSingle")
        self._pot = kwargs["pot"]
        if not "alpha" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckelTriaxial")
        if not "beta" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckelTriaxial")
        
        self._delta = kwargs["delta"]
        
        # Pre-calculate everything
        self._x, self._y, self._z = coords.cyl_to_rect(self._R, self._z, self._phi)
        self._vx, self._vy, self._vz = coords.cyl_to_rect_vec(self._vR, self._vT, self._vz, self._phi)

        self._alpha, self._beta = estimateAlpbaBetaStaeckelTriaxial(self._pot, self._R, self._z, self._phi)
        self._l, self._m, self._n = cartesian_to_ellipsoidal(self._x, self._y, self._z, self._alpha, self._beta, -1)
        self._vl, self._vm, self._vn = cartesian_to_ellipsoidal_vect(self._x, self._y, self._z, self._vx, self._vy, self._vz, self._alpha, self._beta, -1)

        self._E = self.calcE()
        self._evalPotInit = _evaluatePotentials(self._pot, self._R, self._z, self._phi)
        #get A
        #get B
        
        #get tau plus, minus for each
        
        #perform numerical integration on the integrand
        
        return None

    def angleR(self, **kwargs):
        raise NotImplementedError(
            "'angleR' not yet implemented for Staeckel approximation"
        )

    def TR(self, **kwargs):
        raise NotImplementedError("'TR' not implemented yet for Staeckel approximation")

    def Tphi(self, **kwargs):
        raise NotImplementedError(
            "'Tphi' not implemented yet for Staeckel approxximation"
        )

    def I(self, **kwargs):
        raise NotImplementedError("'I' not implemented yet for Staeckel approxximation")

    def Jphi(self):  # pragma: no cover
        pass

    def JR(self, **kwargs):
        pass

    def Jz(self, **kwargs):
        pass

    def calcE(self, **kwargs):
        return self._evalPotInit + self._vx**2.0 / 2.0 + self._vy**2.0 / 2.0 + self._vz**2.0 / 2.0

    def _chi_l(self, l, m, n, pot):
        return (l - m)*(n - l) * pot

    def _chi_m(self, l, m, n, pot):
        return (m - n)*(l - m) * pot

    def _chi_n(self, l, m, n, pot):
        return (n - l)*(m - n) * pot

    def _calculate_A_B(self, alpha, beta, gamma):

        P_l2 = ((self._l - self._m) * (self._l - self._n)) / (4 * (self._l + alpha) * (self._l + beta) * (self._l + gamma))
        P_m2 = ((self._m - self._n) * (self._m - self._l)) / (4 * (self._m + alpha) * (self._m + beta) * (self._m + gamma))
        P_n2 = ((self._n - self._m) * (self._n - self._l)) / (4 * (self._n + alpha) * (self._n + beta) * (self._n + gamma))

        A_l = (self._m + self._n) * self._E + (self._l - self._m)*(self._vm ** 2)/(2*P_m2) + (self._l - self._n)*(self._vn ** 2)/(2*P_n2)
        A_m = (self._l + self._n) * self._E + (self._m - self._l)*(self._vl ** 2)/(2*P_l2) + (self._m - self._n)*(self._vn ** 2)/(2*P_n2)
        A_n = (self._l + self._m) * self._E + (self._n - self._l)*(self._vl ** 2)/(2*P_l2) + (self._n - self._m)*(self._vm ** 2)/(2*P_m2)
        
        chi_l = self._chi_l(self._l, self._m, self._n, self._evalPotInit)
        chi_m = self._chi_m(self._l, self._m, self._n, self._evalPotInit)
        chi_n = self._chi_n(self._l, self._m, self._n, self._evalPotInit)

        B_l = 2*(self._l + alpha)*(self._l + beta)*(self._l + gamma)*self._vl**2 - (self._l**2)*self._E + self._l * A_l - chi_l
        B_m = 2*(self._m + alpha)*(self._m + beta)*(self._m + gamma)*self._vm**2 - (self._m**2)*self._E + self._m * A_m - chi_m
        B_n = 2*(self._n + alpha)*(self._n + beta)*(self._n + gamma)*self._vn**2 - (self._n**2)*self._E + self._n * A_n - chi_n

        return A_l, A_m, A_n, B_l, B_m, B_n

    def calcTauPlusTauMinus(self, **kwargs):
        """
        Calculate the tau +, tau -

        In paper, find the roots of equation 12 (or, perhaps better expressed, equation 17)

        Important note:
        The LHS has terms (tau + alpha) * (tau + beta) * (tau + gamma)
        Recall the parameter constraints -gamma <= nu <= -beta <= mu <= -alpha <= lambda
        Therefore, 
            tau = lambda, LHS = (+) * (+) * (+) = +
            tau = mu, LHS = (-) * (+) * (+) = -
            tau = nu, LHS = (-) * (-) * (+) = +
        Sign effects are included in root solving below.
            
        Returns
        -------
        tuple
            (taup,taum)

        Notes
        -----
        - 2025-04-07 - Written - Weatherall
        """
        if hasattr(self, "_tauptuam"):  # pragma: no cover
            return self._tauptuam
        E = self._E
        
        return None

def cartesian_to_ellipsoidal(x,y,z,alpha,beta,gamma):
    #using the code from galaxiesbook.org
    x= numpy.atleast_1d(x)
    y= numpy.atleast_1d(y)
    z= numpy.atleast_1d(z)
    N= len(x)
    out= numpy.empty((N,3))
    for ii,(tx,ty,tz) in enumerate(zip(x,y,z)):
        these_coords= numpy.polynomial.polynomial.Polynomial(\
                        (beta*gamma*tx**2.+alpha*gamma*ty**2.+alpha*beta*tz**2.
                         -alpha*beta*gamma,
                        (beta+gamma)*tx**2.+(alpha+gamma)*ty**2.+(alpha+beta)*tz**2.
                         -alpha*beta-alpha*gamma-beta*gamma,
                        tx**2.+ty**2.+tz**2.-alpha-beta-gamma,
                        -1.)).roots()
        out[ii]= sorted(these_coords)[::-1]
    return out

def cartesian_to_ellipsoidal_vect(x, y, z, vx, vy, vz, alpha, beta, gamma):
    #using equation 4, derived from 2
    l, m, n = cartesian_to_ellipsoidal(x, y, z, alpha, beta, gamma)
    
    #equation 2
    x2 = ((l + alpha) * (m + alpha) * (n + alpha)) / ((alpha - beta) * (alpha - gamma))
    y2 = ((l + beta) * (m + beta) * (n + beta)) / ((beta - alpha) * (beta - gamma))
    z2 = ((l + gamma) * (m + gamma) * (n + gamma)) / ((gamma - beta) * (gamma - alpha))

    vl = (vx/2) * numpy.sqrt(x2 / ((l + alpha) ** 2)) \
        + (vy/2) * numpy.sqrt(y2 / ((l + beta) ** 2)) \
        + (vz/2) * numpy.sqrt(z2 / ((l + gamma) ** 2))
    
    vm = (vx/2) * numpy.sqrt(x2 / ((m + alpha) ** 2)) \
        + (vy/2) * numpy.sqrt(y2 / ((m + beta) ** 2)) \
        + (vz/2) * numpy.sqrt(z2 / ((m + gamma) ** 2))
    
    vn = (vx/2) * numpy.sqrt(x2 / ((n + alpha) ** 2)) \
        + (vy/2) * numpy.sqrt(y2 / ((n + beta) ** 2)) \
        + (vz/2) * numpy.sqrt(z2 / ((n + gamma) ** 2))

    return (vl, vm, vn)

def ellipsoidal_to_cartesian(l,m,n,alpha,beta,gamma):
    x= numpy.sqrt((l+alpha)*(m+alpha)*(n+alpha)/(alpha-beta)/(alpha-gamma))
    y= numpy.sqrt((l+beta)*(m+beta)*(n+beta)/(beta-alpha)/(beta-gamma))
    z= numpy.sqrt((l+gamma)*(m+gamma)*(n+gamma)/(gamma-beta)/(gamma-alpha))
    return numpy.array([x,y,z]).T

@potential_physical_input
@physical_conversion("position", pop=True)
def estimateAlpbaBetaStaeckelTriaxial(pot, R, z, phi):
    """
    Estimate values for alpha, beta using the closed loop estimate technique in Sanders & Binney (2015)

    Parameters
    ----------
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
    R : float or numpy.ndarray
        coordinates
    z : float or numpy.ndarray
        coordinates
    phi : float or numpy.ndarray
        coordinates

    Returns
    -------
    float or numpy.ndarray
        estimate of alpha, beta

    Notes
    -----
    - 2026-04-07 - written - Weatherall
    """

    pot = _check_potential_list_and_deprecate(pot)
    if _isNonAxi(pot):
        raise PotentialError(
            "Calling estimateAlpbaBetaStaeckelTriaxial with non-axisymmetric potentials is not supported"
        )
    
    alpha = 0.5
    beta = 1

    return alpha, beta
