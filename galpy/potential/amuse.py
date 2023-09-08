# galpy.potential.amuse: AMUSE representation of galpy potentials
import numpy
from amuse.support.literature import LiteratureReferencesMixIn
from amuse.units import units
from amuse.units.quantities import ScalarQuantity

from .. import potential
from ..util import conversion


class galpy_profile(LiteratureReferencesMixIn):
    """
    User-defined potential from galpy

    .. [#] Bovy, J, 2015, galpy: A Python Library for Galactic Dynamics, Astrophys. J. Supp. 216, 29 [2015ApJS..216...29B]

    """

    def __init__(self, pot, t=0.0, tgalpy=0.0, ro=8, vo=220.0, reverse=False):
        """
        Initialize a galpy potential for use with AMUSE.

        Parameters
        ----------
        pot : Potential instance or list thereof, optional
            Potential object(s) to be used with AMUSE.
        t : float or Quantity, optional
            Start time for AMUSE simulation (can be an AMUSE Quantity).
        tgalpy : float or Quantity, optional
            Start time for galpy potential, can be less than zero (can be Quantity).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        reverse : bool, optional
            Set whether galpy potential evolves forwards or backwards in time (default: False).

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)

        """
        LiteratureReferencesMixIn.__init__(self)
        self.pot = pot
        self.ro = ro
        self.vo = vo
        self.reverse = reverse
        # Initialize model time
        if isinstance(t, ScalarQuantity):
            self.model_time = t
        else:
            self.model_time = (
                t * conversion.time_in_Gyr(ro=self.ro, vo=self.vo) | units.Gyr
            )
        # Initialize galpy time
        if isinstance(tgalpy, ScalarQuantity):
            self.tgalpy = tgalpy.value_in(units.Gyr) / conversion.time_in_Gyr(
                ro=self.ro, vo=self.vo
            )
        else:
            self.tgalpy = tgalpy

    def evolve_model(self, time):
        """
        Evolve time parameters to t_end.

        Parameters
        ----------
        time : float
            End time for the potential evolution.

        Returns
        -------
        None

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)

        """
        dt = time - self.model_time
        self.model_time = time
        if self.reverse:
            self.tgalpy -= dt.value_in(units.Gyr) / conversion.time_in_Gyr(
                ro=self.ro, vo=self.vo
            )
        else:
            self.tgalpy += dt.value_in(units.Gyr) / conversion.time_in_Gyr(
                ro=self.ro, vo=self.vo
            )

    def get_potential_at_point(self, eps, x, y, z):
        """
        Get potential at a given location in the potential.

        Parameters
        ----------
        eps : AMUSE Quantity
            Softening length (necessary for AMUSE, but not used by galpy potential).
        x,y,z : AMUSE Quantity
            Position in the potential.

        Returns
        -------
        AMUSE Quantity
            Phi(x,y,z).

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        - 2019-11-06 - Added physical compatibility - Starkman (UofT).

        """
        R = numpy.sqrt(x.value_in(units.kpc) ** 2.0 + y.value_in(units.kpc) ** 2.0)
        zed = z.value_in(units.kpc)
        phi = numpy.arctan2(y.value_in(units.kpc), x.value_in(units.kpc))
        res = potential.evaluatePotentials(
            self.pot,
            R / self.ro,
            zed / self.ro,
            phi=phi,
            t=self.tgalpy,
            ro=self.ro,
            vo=self.vo,
            use_physical=False,
        )
        return res * self.vo**2 | units.kms**2

    def get_gravity_at_point(self, eps, x, y, z):
        """
        Get acceleration due to potential at a given location in the potential.

        Parameters
        ----------
        eps : AMUSE Quantity
            Softening length (necessary for AMUSE, but not used by galpy potential).
        x,y,z : AMUSE Quantity
            Position in the potential.

        Returns
        -------
        ax : AMUSE Quantity
            Acceleration in the x-direction.
        ay : AMUSE Quantity
            Acceleration in the y-direction.
        az : AMUSE Quantity
            Acceleration in the z-direction.

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        - 2019-11-06 - Added physical compatibility - Starkman (UofT).

        """
        R = numpy.sqrt(x.value_in(units.kpc) ** 2.0 + y.value_in(units.kpc) ** 2.0)
        zed = z.value_in(units.kpc)
        phi = numpy.arctan2(y.value_in(units.kpc), x.value_in(units.kpc))
        # Cylindrical force
        Rforce = potential.evaluateRforces(
            self.pot,
            R / self.ro,
            zed / self.ro,
            phi=phi,
            t=self.tgalpy,
            use_physical=False,
        )
        phitorque = potential.evaluatephitorques(
            self.pot,
            R / self.ro,
            zed / self.ro,
            phi=phi,
            t=self.tgalpy,
            use_physical=False,
        ) / (R / self.ro)
        zforce = potential.evaluatezforces(
            self.pot,
            R / self.ro,
            zed / self.ro,
            phi=phi,
            t=self.tgalpy,
            use_physical=False,
        )
        # Convert cylindrical force --> rectangular
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        ax = (Rforce * cp - phitorque * sp) * conversion.force_in_kmsMyr(
            ro=self.ro, vo=self.vo
        ) | units.kms / units.Myr
        ay = (Rforce * sp + phitorque * cp) * conversion.force_in_kmsMyr(
            ro=self.ro, vo=self.vo
        ) | units.kms / units.Myr
        az = (
            zforce * conversion.force_in_kmsMyr(ro=self.ro, vo=self.vo)
            | units.kms / units.Myr
        )
        return ax, ay, az

    def mass_density(self, x, y, z):
        """
        Get mass density at a given location in the potential

        Parameters
        ----------
        eps : AMUSE Quantity
            Softening length (necessary for AMUSE, but not used by galpy potential)
        x,y,z : AMUSE Quantity
            Position in the potential

        Returns
        -------
        AMUSE Quantity
            The density

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        - 2019-11-06 - added physical compatibility - Starkman (UofT)

        """
        R = numpy.sqrt(x.value_in(units.kpc) ** 2.0 + y.value_in(units.kpc) ** 2.0)
        zed = z.value_in(units.kpc)
        phi = numpy.arctan2(y.value_in(units.kpc), x.value_in(units.kpc))
        res = potential.evaluateDensities(
            self.pot,
            R / self.ro,
            zed / self.ro,
            phi=phi,
            t=self.tgalpy,
            ro=self.ro,
            vo=self.vo,
            use_physical=False,
        ) * conversion.dens_in_msolpc3(self.vo, self.ro)
        return res | units.MSun / units.parsec**3

    def circular_velocity(self, r):
        """
        Get circular velocity at a given radius in the potential

        Parameters
        ----------
        r : AMUSE Quantity
            Radius in the potential

        Returns
        -------
        AMUSE Quantity
            The circular velocity

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        - 2019-11-06 - added physical compatibility - Starkman (UofT)

        """
        res = potential.vcirc(
            self.pot,
            r.value_in(units.kpc) / self.ro,
            phi=0,
            t=self.tgalpy,
            ro=self.ro,
            vo=self.vo,
            use_physical=False,
        )
        return res * self.vo | units.kms

    def enclosed_mass(self, r):
        """
        Get mass enclosed within a given radius in the potential

        Parameters
        ----------
        r : AMUSE Quantity
            Radius in the potential

        Returns
        -------
        AMUSE Quantity
            The mass enclosed

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        - 2019-11-06 - added physical compatibility - Starkman (UofT)

        """
        vc = (
            potential.vcirc(
                self.pot,
                r.value_in(units.kpc) / self.ro,
                phi=0,
                t=self.tgalpy,
                ro=self.ro,
                vo=self.vo,
                use_physical=False,
            )
            * self.vo
        )
        return (vc**2.0) * r.value_in(units.parsec) / conversion._G | units.MSun

    def stop(self):
        """
        Stop the potential model (necessary function for AMUSE)

        Notes
        -----
        - 2019-08-12 - Written - Webb (UofT)
        """
        pass
