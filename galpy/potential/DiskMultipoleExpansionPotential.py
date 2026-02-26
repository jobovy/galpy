###############################################################################
#   DiskMultipoleExpansionPotential.py: Potential expansion for disk+halo
#   potentials using MultipoleExpansionPotential
###############################################################################
import numpy

from .KuijkenDubinskiDiskExpansionPotential import (
    KuijkenDubinskiDiskExpansionPotential,
)
from .MultipoleExpansionPotential import MultipoleExpansionPotential


class DiskMultipoleExpansionPotential(KuijkenDubinskiDiskExpansionPotential):
    """Class that implements a basis-function-expansion technique for solving the Poisson equation for disk (+halo) systems. We solve the Poisson equation for a given density :math:`\\rho(R,\\phi,z)` by introducing *K* helper function pairs :math:`[\\Sigma_i(R),h_i(z)]`, with :math:`h_i(z) = \\mathrm{d}^2 H(z) / \\mathrm{d} z^2` and search for solutions of the form

        .. math::

           \\Phi(R,\\phi,z = \\Phi_{\\mathrm{ME}}(R,\\phi,z) + 4\\pi G\\sum_i \\Sigma_i(r)\\,H_i(z)\\,,

    where :math:`r` is the spherical radius :math:`r^2 = R^2+z^2`. We can solve for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)` by solving

        .. math::

           \\frac{\\Delta \\Phi_{\\mathrm{ME}}(R,\\phi,z)}{4\\pi G} = \\rho(R,\\phi,z) - \\sum_i\\left\\{ \\Sigma_i(r)\\,h_i(z) + \\frac{\\mathrm{d}^2 \\Sigma_i(r)}{\\mathrm{d} r^2}\\,H_i(z)+\\frac{2}{r}\\,\\frac{\\mathrm{d} \\Sigma_i(r)}{\\mathrm{d} r}\\left[H_i(z)+z\\,\\frac{\\mathrm{d}H_i(z)}{\\mathrm{d} z}\\right]\\right\\}\\,.

    We solve this equation by using the :ref:`MultipoleExpansionPotential <multipole_expansion_potential>` class. This technique works very well if the disk portion of the potential can be exactly written as :math:`\\rho_{\\mathrm{disk}} = \\sum_i \\Sigma_i(R)\\,h_i(z)`, because the effective density on the right-hand side of this new Poisson equation is then not 'disky' and can be well represented using spherical harmonics. But the technique is general and can be used to compute the potential of any disk+halo potential; the closer the disk is to :math:`\\rho_{\\mathrm{disk}} \\approx \\sum_i \\Sigma_i(R)\\,h_i(z)`, the better the technique works.

    This technique was introduced by `Kuijken & Dubinski (1995) <http://adsabs.harvard.edu/abs/1995MNRAS.277.1341K>`__ and was popularized by `Dehnen & Binney (1998) <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__. The current implementation is a slight generalization of the technique in those papers and uses the :ref:`MultipoleExpansionPotential <multipole_expansion_potential>` to solve the Poisson equation for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)`.

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
        L=10,
        rgrid=numpy.geomspace(1e-3, 30, 1001),
        symmetry=None,
        costheta_order=None,
        phi_order=None,
        k=5,
        ro=None,
        vo=None,
    ):
        """
        Initialize a DiskMultipoleExpansionPotential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1); cannot have units currently.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        dens : callable
            Function of R,z[,phi optional] that gives the density [in natural units, cannot return a Quantity currently].
        L : int, optional
            Maximum spherical harmonic degree + 1 (l goes from 0 to L-1).
        rgrid : numpy.ndarray, optional
            Radial grid points (1D array). Default: ``numpy.geomspace(1e-3, 30, 1001)``.
        symmetry : str or None, optional
            ``'spherical'``, ``'axisymmetric'``, or ``None`` (general). If None and the density is axisymmetric, ``'axisymmetric'`` is used automatically.
        costheta_order : int, optional
            Gauss-Legendre quadrature order for theta.
        phi_order : int, optional
            Number of uniform phi points for trapezoidal rule.
        k : int, optional
            Spline interpolation degree for radial functions (default: 5).
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
        - 2026-02-22 - Written - Bovy (UofT)

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
        # Auto-detect symmetry if not specified
        if symmetry is None and not self.isNonAxi:
            symmetry = "axisymmetric"
        self._me = MultipoleExpansionPotential(
            amp=1.0,
            dens=self._phiME_dens_func,
            L=L,
            rgrid=rgrid,
            symmetry=symmetry,
            costheta_order=costheta_order,
            phi_order=phi_order,
            k=k,
            ro=None,
            vo=None,
        )
        self._finish_init(normalize)
        return None
