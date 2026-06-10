###############################################################################
#   DiskSCFPotential.py: Potential expansion for disk+halo potentials
###############################################################################
import numpy

from ..util import conversion
from .KuijkenDubinskiDiskExpansionPotential import (
    KuijkenDubinskiDiskExpansionPotential,
)
from .SCFPotential import SCFPotential


class DiskSCFPotential(KuijkenDubinskiDiskExpansionPotential):
    """Class that implements a basis-function-expansion technique for solving the Poisson equation for disk (+halo) systems. We solve the Poisson equation for a given density :math:`\\rho(R,\\phi,z)` by introducing *K* helper function pairs :math:`[\\Sigma_i(R),h_i(z)]`, with :math:`h_i(z) = \\mathrm{d}^2 H(z) / \\mathrm{d} z^2` and search for solutions of the form

        .. math::

           \\Phi(R,\\phi,z = \\Phi_{\\mathrm{ME}}(R,\\phi,z) + 4\\pi G\\sum_i \\Sigma_i(r)\\,H_i(z)\\,,

    where :math:`r` is the spherical radius :math:`r^2 = R^2+z^2`. We can solve for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)` by solving

        .. math::

           \\frac{\\Delta \\Phi_{\\mathrm{ME}}(R,\\phi,z)}{4\\pi G} = \\rho(R,\\phi,z) - \\sum_i\\left\\{ \\Sigma_i(r)\\,h_i(z) + \\frac{\\mathrm{d}^2 \\Sigma_i(r)}{\\mathrm{d} r^2}\\,H_i(z)+\\frac{2}{r}\\,\\frac{\\mathrm{d} \\Sigma_i(r)}{\\mathrm{d} r}\\left[H_i(z)+z\\,\\frac{\\mathrm{d}H_i(z)}{\\mathrm{d} z}\\right]\\right\\}\\,.

    We solve this equation by using the :ref:`SCFPotential <scf_potential>` class and methods (:ref:`scf_compute_coeffs_axi <scf_compute_coeffs_axi>` or :ref:`scf_compute_coeffs <scf_compute_coeffs>` depending on whether :math:`\\rho(R,\\phi,z)` is axisymmetric or not). This technique works very well if the disk portion of the potential can be exactly written as :math:`\\rho_{\\mathrm{disk}} = \\sum_i \\Sigma_i(R)\\,h_i(z)`, because the effective density on the right-hand side of this new Poisson equation is then not 'disky' and can be well represented using spherical harmonics. But the technique is general and can be used to compute the potential of any disk+halo potential; the closer the disk is to :math:`\\rho_{\\mathrm{disk}} \\approx \\sum_i \\Sigma_i(R)\\,h_i(z)`, the better the technique works.

    This technique was introduced by `Kuijken & Dubinski (1995) <http://adsabs.harvard.edu/abs/1995MNRAS.277.1341K>`__ and was popularized by `Dehnen & Binney (1998) <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__. The current implementation is a slight generalization of the technique in those papers and uses the SCF approach of `Hernquist & Ostriker (1992)
    <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`__ to solve the Poisson equation for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)` rather than solving it on a grid using spherical harmonics and interpolating the solution (as done in `Dehnen & Binney 1998 <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__).

    """

    def __init__(
        self,
        amp=1.0,
        normalize=False,
        dens=lambda R, z: 13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z)),
        Sigma={"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
        hz={"type": "exp", "h": 1.0 / 27.0},
        Sigma_amp=None,
        dSigmadR=None,
        d2SigmadR2=None,
        Hz=None,
        dHzdz=None,
        N=10,
        L=10,
        a=1.0,
        radial_order=None,
        costheta_order=None,
        phi_order=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a DiskSCFPotential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1); cannot have units currently.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        dens : callable
            Function of R,z[,phi optional] that gives the density [in natural units, cannot return a Quantity currently].
        N : int, optional
            Number of radial basis functions to use in the SCF expansion.
        L : int, optional
            Number of angular basis functions to use in the SCF expansion.
        a : float or Quantity, optional
            Scale radius for the SCF expansion.
        radial_order : int, optional
            Order of the radial basis functions to use in the SCF expansion.
        costheta_order : int, optional
            Order of the angular basis functions to use in the SCF expansion.
        phi_order : int, optional
            Order of the azimuthal basis functions to use in the SCF expansion.
        Sigma : dict or callable
            Either a dictionary of surface density (example: {'type':'exp','h':1./3.,'amp':1.,'Rhole':0.} for amp x exp(-Rhole/R-R/h) ) or a function of R that gives the surface density.
        hz : dict or callable
            Either a dictionary of vertical profile, either 'exp' or 'sech2' (example {'type':'exp','h':1./27.} for exp(-|z|/h)/[2h], sech2 is sech^2(z/[2h])/[4h]) or a function of z that gives the vertical profile.
        Sigma_amp : float, optional
            Amplitude to apply to all Sigma functions.
        dSigmadR : callable, optional
            Function that gives d Sigma / d R.
        d2SigmadR2 : callable, optional
            Function that gives d^2 Sigma / d R^2.
        Hz : callable, optional
            Function of z such that d^2 Hz(z) / d z^2 = hz.
        dHzdz : callable, optional
            Function of z that gives d Hz(z) / d z.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Either specify (Sigma,hz) or (Sigma_amp,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz)
        - The built-in dict-specified Sigma/hz profiles are backend-agnostic (numpy/jax/torch); for jax/torch evaluation, any *user-provided* Sigma/dSigmadR/d2SigmadR2/hz/Hz/dHzdz callables must accept backend arrays (e.g., be written with ``galpy.backend.get_namespace``)
        - Written - Bovy (UofT) - 2016-12-26

        """
        KuijkenDubinskiDiskExpansionPotential.__init__(
            self,
            amp=amp,
            dens=dens,
            Sigma=Sigma,
            hz=hz,
            Sigma_amp=Sigma_amp,
            dSigmadR=dSigmadR,
            d2SigmadR2=d2SigmadR2,
            Hz=Hz,
            dHzdz=dHzdz,
            ro=ro,
            vo=vo,
        )
        a = conversion.parse_length(a, ro=self._ro)
        # Auto-detect symmetry if not specified
        symmetry = "axisymmetric" if not self.isNonAxi else None
        self._me = SCFPotential.from_density(
            dens=self._phiME_dens_func,
            N=N,
            L=L,
            a=a,
            symmetry=symmetry,
            radial_order=radial_order,
            costheta_order=costheta_order,
            phi_order=phi_order,
            ro=None,
            vo=None,
        )
        self._finish_init(normalize)
        return None
