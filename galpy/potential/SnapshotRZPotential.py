import copy
import hashlib
from os import system

import numpy
from scipy import interpolate

from ..util._optional_deps import _PYNBODY_LOADED
from .interpRZPotential import (
    calc_2dsplinecoeffs_c,
    interpRZPotential,
    scalarVectorDecorator,
    zsymDecorator,
)
from .Potential import Potential

if _PYNBODY_LOADED:
    import pynbody
    from pynbody import gravity
    from pynbody.units import NoUnit


class SnapshotRZPotential(Potential):
    """Class that implements an axisymmetrized version of the potential of an N-body snapshot (requires `pynbody <http://pynbody.github.io>`__)

    `_evaluate`, `_Rforce`, and `_zforce` calculate a hash for the
    array of points that is passed in by the user. The hash and
    corresponding potential/force arrays are stored -- if a subsequent
    request matches a previously computed hash, the previous results
    are returned and not recalculated.
    """

    def __init__(self, s, num_threads=None, nazimuths=4, ro=None, vo=None):
        """
        Initialize a SnapshotRZ potential object

        Parameters
        ----------
        s : pynbody.snapshot
            A simulation snapshot loaded with pynbody.
        num_threads : int, optional
            Number of threads to use for calculation. Default is None.
        nazimuths : int, optional
            Number of azimuths to average over. Default is 4.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013 - Written - Rok Roskar (ETH)
        - 2014-11-24 - Edited for merging into main galpy - Bovy (IAS)

        """
        if not _PYNBODY_LOADED:  # pragma: no cover
            raise ImportError(
                "The SnapShotRZPotential class is designed to work with pynbody snapshots, which cannot be loaded (probably because it is not installed) -- obtain from pynbody.github.io"
            )
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
        self._s = s
        self._point_hash = {}
        if num_threads is None:
            self._num_threads = pynbody.config["number_of_threads"]
        else:
            self._num_threads = num_threads
        # Set up azimuthal averaging
        self._naz = nazimuths
        self._cosaz = numpy.cos(
            numpy.arange(self._naz, dtype="float") / self._naz * 2.0 * numpy.pi
        )
        self._sinaz = numpy.sin(
            numpy.arange(self._naz, dtype="float") / self._naz * 2.0 * numpy.pi
        )
        self._zones = numpy.ones(self._naz)
        self._zzeros = numpy.zeros(self._naz)
        return None

    @scalarVectorDecorator
    def _evaluate(self, R, z, phi=None, t=None, dR=None, dphi=None):
        pot, acc = self._setup_potential(R, z)
        return pot

    @scalarVectorDecorator
    def _Rforce(self, R, z, phi=None, t=None, dR=None, dphi=None):
        pot, acc = self._setup_potential(R, z)
        return acc[:, 0]

    @scalarVectorDecorator
    def _zforce(self, R, z, phi=None, t=None, dR=None, dphi=None):
        pot, acc = self._setup_potential(R, z)
        return acc[:, 1]

    def _setup_potential(self, R, z, use_pkdgrav=False):
        # compute the hash for the requested grid
        new_hash = hashlib.md5(numpy.array([R, z])).hexdigest()

        # if we computed for these points before, return; otherwise compute
        if new_hash in self._point_hash:
            pot, rz_acc = self._point_hash[new_hash]

        #        if use_pkdgrav :

        else:
            # set up the four points per R,z pair to mimic axisymmetry
            points = numpy.zeros((len(R), self._naz, 3))

            for i in range(len(R)):
                points[i] = numpy.array(
                    [R[i] * self._cosaz, R[i] * self._sinaz, z[i] * self._zones]
                ).T

            points_new = points.reshape(points.size // 3, 3)
            pot, acc = gravity.calc.direct(
                self._s, points_new, num_threads=self._num_threads
            )

            pot = pot.reshape(len(R), self._naz)
            acc = acc.reshape(len(R), self._naz, 3)

            # need to average the potentials
            pot = pot.mean(axis=1)

            # get the radial accelerations
            rz_acc = numpy.zeros((len(R), 2))
            rvecs = numpy.array([self._cosaz, self._sinaz, self._zzeros]).T

            for i in range(len(R)):
                for j, rvec in enumerate(rvecs):
                    rz_acc[i, 0] += acc[i, j].dot(rvec)
                    rz_acc[i, 1] += acc[i, j, 2]
            rz_acc /= self._naz

            # store the computed values for reuse
            self._point_hash[new_hash] = [pot, rz_acc]

        return pot, rz_acc


class InterpSnapshotRZPotential(interpRZPotential):
    """
    Interpolated axisymmetrized potential extracted from a simulation output (see ``interpRZPotential`` and ``SnapshotRZPotential``)
    """

    def __init__(
        self,
        s,
        ro=None,
        vo=None,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        interpepifreq=False,
        interpverticalfreq=False,
        interpPot=True,
        enable_c=True,
        logR=True,
        zsym=True,
        numcores=None,
        nazimuths=4,
        use_pkdgrav=False,
    ):
        """
        Initialize an InterpSnapshotRZPotential instance

        Parameters
        ----------
        s : pynbody.snapshot
            A simulation snapshot loaded with pynbody.
        rgrid : tuple, optional
            R grid to be given to linspace as in rs= linspace(*rgrid).
        zgrid : tuple, optional
            z grid to be given to linspace as in zs= linspace(*zgrid).
        interpepifreq : bool, optional
            If True, interpolate the epicycle frequencies (default: False).
        interpverticalfreq : bool, optional
            If True, interpolate the vertical frequencies (default: False).
        interpPot : bool, optional
            If True, interpolate the potential (default: True).
        enable_c : bool, optional
            If True, use C for the interpolation (default: True).
        logR : bool, optional
            If True, rgrid is in the log of R so logrs= linspace(*rgrid) (default: True).
        zsym : bool, optional
            If True (default), the potential is assumed to be symmetric around z=0 (so you can use, e.g.,  zgrid=(0.,1.,101)).
        numcores : int, optional
            Number of cores to use for the interpolation (default: from pynbody configuration).
        nazimuths : int, optional
            Number of azimuths to average over (default: 4).
        use_pkdgrav : bool, optional
            If True, use PKDGRAV to calculate the snapshot's potential and forces (default: False).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013 - Written - Rok Roskar (ETH)
        - 2014-11-24 - Edited for merging into main galpy - Bovy (IAS)
        """
        if not _PYNBODY_LOADED:  # pragma: no cover
            raise ImportError(
                "The InterpSnapRZShotPotential class is designed to work with pynbody snapshots, which cannot be loaded (probably because it is not installed) -- obtain from pynbody.github.io"
            )

        # initialize using the base class
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)

        # other properties
        if numcores is None:
            self._numcores = pynbody.config["number_of_threads"]
        else:
            self._numcores = numcores
        self._s = s

        # Set up azimuthal averaging
        self._naz = nazimuths
        self._cosaz = numpy.cos(
            numpy.arange(self._naz, dtype="float") / self._naz * 2.0 * numpy.pi
        )
        self._sinaz = numpy.sin(
            numpy.arange(self._naz, dtype="float") / self._naz * 2.0 * numpy.pi
        )
        self._zones = numpy.ones(self._naz)
        self._zzeros = numpy.zeros(self._naz)

        # the interpRZPotential class sets these flags
        self._enable_c = enable_c
        self.hasC = True

        # set up the flags for interpolated quantities
        # since the potential and force are always calculated together,
        # set the force interpolations to true if potential is true and
        # vice versa
        self._interpPot = interpPot
        self._interpRforce = self._interpPot
        self._interpzforce = self._interpPot
        self._interpvcirc = self._interpPot

        # these require additional calculations so set them separately
        self._interpepifreq = interpepifreq
        self._interpverticalfreq = interpverticalfreq

        # make the potential accessible at points beyond the grid
        self._origPot = SnapshotRZPotential(s, self._numcores)

        # setup the grid
        self._zsym = zsym
        self._logR = logR

        self._rgrid = numpy.linspace(*rgrid)
        if logR:
            self._rgrid = numpy.exp(self._rgrid)
            self._logrgrid = numpy.log(self._rgrid)
            rs = self._logrgrid
        else:
            rs = self._rgrid

        self._zgrid = numpy.linspace(*zgrid)

        # calculate the grids
        self._setup_potential(self._rgrid, self._zgrid, use_pkdgrav=use_pkdgrav)

        if enable_c and interpPot:
            self._potGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._potGrid)
            self._rforceGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._rforceGrid)
            self._zforceGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._zforceGrid)

        else:
            self._potInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._potGrid, kx=3, ky=3, s=0.0
            )
            self._rforceInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._rforceGrid, kx=3, ky=3, s=0.0
            )
            self._zforceInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._zforceGrid, kx=3, ky=3, s=0.0
            )
        if interpepifreq:
            self._R2interp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._R2derivGrid, kx=3, ky=3, s=0.0
            )

        if interpverticalfreq:
            self._z2interp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._z2derivGrid, kx=3, ky=3, s=0.0
            )

        if interpepifreq and interpverticalfreq:
            self._Rzinterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._RzderivGrid, kx=3, ky=3, s=0.0
            )

        # setup the derived quantities
        if interpPot:
            self._vcircGrid = numpy.sqrt(self._rgrid * (-self._rforceGrid[:, 0]))
            self._vcircInterp = interpolate.InterpolatedUnivariateSpline(
                rs, self._vcircGrid, k=3
            )

        if interpepifreq:
            self._epifreqGrid = numpy.sqrt(
                self._R2derivGrid[:, 0] - 3.0 / self._rgrid * self._rforceGrid[:, 0]
            )
            goodindx = True ^ numpy.isnan(self._epifreqGrid)
            self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                rs[goodindx], self._epifreqGrid[goodindx], k=3
            )
            self._epigoodindx = goodindx
        if interpverticalfreq:
            self._verticalfreqGrid = numpy.sqrt(numpy.abs(self._z2derivGrid[:, 0]))
            goodindx = True ^ numpy.isnan(self._verticalfreqGrid)
            self._verticalfreqInterp = interpolate.InterpolatedUnivariateSpline(
                rs[goodindx], self._verticalfreqGrid[goodindx], k=3
            )
            self._verticalgoodindx = goodindx

    def _setup_potential(self, R, z, use_pkdgrav=False, dr=0.0001):
        """

        Calculates the potential and force grids for the snapshot for
        use with other galpy functions.

        **Input**:

        *R*: R grid coordinates (numpy array)

        *z*: z grid coordinates (numpy array)

        **Optional Keywords**:

        *use_pkdgrav*: (False) whether to use pkdgrav for the gravity
         calculation

        *dr*: (0.01) offset to use for the gradient calculation - the
         points are positioned at +/- dr from the central point

        """
        # set up the four points per R,z pair to mimic axisymmetry
        points = numpy.zeros((len(R), len(z), self._naz, 3))

        for i in range(len(R)):
            for j in range(len(z)):
                points[i, j] = numpy.array(
                    [R[i] * self._cosaz, R[i] * self._sinaz, z[j] * self._zones]
                ).T

        points_new = points.reshape(points.size // 3, 3)
        self.points = points_new

        # set up the points to calculate the second derivatives
        zgrad_points = numpy.zeros((len(points_new) * 2, 3))
        rgrad_points = numpy.zeros((len(points_new) * 2, 3))
        for i, p in enumerate(points_new):
            zgrad_points[i * 2] = p
            zgrad_points[i * 2][2] -= dr
            zgrad_points[i * 2 + 1] = p
            zgrad_points[i * 2 + 1][2] += dr

            rgrad_points[i * 2] = p
            rgrad_points[i * 2][:2] -= p[:2] / numpy.sqrt(numpy.dot(p[:2], p[:2])) * dr
            rgrad_points[i * 2 + 1] = p
            rgrad_points[i * 2 + 1][:2] += (
                p[:2] / numpy.sqrt(numpy.dot(p[:2], p[:2])) * dr
            )

        if use_pkdgrav:  # pragma: no cover
            raise RuntimeError("using pkdgrav not currently implemented")
            sn = pynbody.snapshot._new(
                len(self._s.d) + len(self._s.g) + len(self._s.s) + len(points_new)
            )
            print("setting up %d grid points" % (len(points_new)))
            # sn['pos'][0:len(self.s)] = self.s['pos']
            # sn['mass'][0:len(self.s)] = self.s['mass']
            # sn['phi'] = 0.0
            # sn['eps'] = 1e3
            # sn['eps'][0:len(self.s)] = self.s['eps']
            # sn['vel'][0:len(self.s)] = self.s['vel']
            # sn['mass'][len(self.s):] = 1e-10
            sn["pos"][len(self._s) :] = points_new
            sn["mass"][len(self._s) :] = 0.0

            sn.write(fmt=pynbody.tipsy.TipsySnap, filename="potgridsnap")
            command = (
                "~/bin/pkdgrav2_pthread -sz %d -n 0 +std -o potgridsnap -I potgridsnap +potout +overwrite %s"
                % (self._numcores, self._s._paramfile["filename"])
            )
            print(command)
            system(command)
            sn = pynbody.load("potgridsnap")
            acc = sn["accg"][len(self._s) :].reshape(len(R) * len(z), self._naz, 3)
            pot = sn["pot"][len(self._s) :].reshape(len(R) * len(z), self._naz)

        else:
            if self._interpPot:
                pot, acc = gravity.calc.direct(
                    self._s, points_new, num_threads=self._numcores
                )

                pot = pot.reshape(len(R) * len(z), self._naz)
                acc = acc.reshape(len(R) * len(z), self._naz, 3)

                # need to average the potentials
                pot = pot.mean(axis=1)

                # get the radial accelerations
                rz_acc = numpy.zeros((len(R) * len(z), 2))
                rvecs = numpy.array([self._cosaz, self._sinaz, self._zzeros]).T

                # reshape the acc to make sure we have a leading index even
                # if we are only evaluating a single point, i.e. we have
                # shape = (1,4,3) not (4,3)
                acc = acc.reshape((len(rz_acc), self._naz, 3))

                for i in range(len(R) * len(z)):
                    for j, rvec in enumerate(rvecs):
                        rz_acc[i, 0] += acc[i, j].dot(rvec)
                        rz_acc[i, 1] += acc[i, j, 2]
                rz_acc /= self._naz

                self._potGrid = pot.reshape((len(R), len(z)))
                self._rforceGrid = rz_acc[:, 0].reshape((len(R), len(z)))
                self._zforceGrid = rz_acc[:, 1].reshape((len(R), len(z)))

            # compute the force gradients

            # first get the accelerations
            if self._interpverticalfreq:
                zgrad_pot, zgrad_acc = gravity.calc.direct(
                    self._s, zgrad_points, num_threads=self._numcores
                )
                # each point from the points used above for pot and acc is straddled by
                # two points to get the gradient across it. Compute the gradient by
                # using a finite difference

                zgrad = numpy.zeros(len(points_new))

                # do a loop through the pairs of points -- reshape the array
                # so that each item is the pair of acceleration vectors
                # then calculate the gradient from the two points
                for i, zacc in enumerate(
                    zgrad_acc.reshape((len(zgrad_acc) // 2, 2, 3))
                ):
                    zgrad[i] = ((zacc[1] - zacc[0]) / (dr * 2.0))[2]

                # reshape the arrays
                self._z2derivGrid = (
                    -zgrad.reshape((len(zgrad) // self._naz, self._naz))
                    .mean(axis=1)
                    .reshape((len(R), len(z)))
                )

            # do the same for the radial component
            if self._interpepifreq:
                rgrad_pot, rgrad_acc = gravity.calc.direct(
                    self._s, rgrad_points, num_threads=self._numcores
                )
                rgrad = numpy.zeros(len(points_new))

                for i, racc in enumerate(
                    rgrad_acc.reshape((len(rgrad_acc) // 2, 2, 3))
                ):
                    point = points_new[i]
                    point[2] = 0.0
                    rvec = point / numpy.sqrt(numpy.dot(point, point))
                    rgrad_vec = (
                        numpy.dot(racc[1], rvec) - numpy.dot(racc[0], rvec)
                    ) / (dr * 2.0)
                    rgrad[i] = rgrad_vec

                self._R2derivGrid = (
                    -rgrad.reshape((len(rgrad) // self._naz, self._naz))
                    .mean(axis=1)
                    .reshape((len(R), len(z)))
                )

            # do the same for the mixed radial-vertical component
            if self._interpepifreq and self._interpverticalfreq:  # reuse this
                Rzgrad = numpy.zeros(len(points_new))
                for i, racc in enumerate(
                    rgrad_acc.reshape((len(rgrad_acc) // 2, 2, 3))
                ):
                    Rzgrad[i] = ((racc[1] - racc[0]) / (dr * 2.0))[2]

                # reshape the arrays
                self._RzderivGrid = (
                    -Rzgrad.reshape((len(Rzgrad) // self._naz, self._naz))
                    .mean(axis=1)
                    .reshape((len(R), len(z)))
                )

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _R2deriv(self, R, Z, phi=0.0, t=0.0):
        return self._R2interp(R, Z)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _z2deriv(self, R, Z, phi=None, t=None):
        return self._z2interp(R, Z)

    @scalarVectorDecorator
    @zsymDecorator(True)
    def _Rzderiv(self, R, Z, phi=None, t=None):
        return self._Rzinterp(R, Z)

    def normalize(self, R0=8.0):
        """

        Normalize all positions by R0 and velocities by Vc(R0).

        If :class:`~scipy.interpolate.RectBivariateSpline` or
        :class:`~scipy.interpolate.InterpolatedUnivariateSpline` are
        used, redefine them for use with the rescaled coordinates.

        To undo the normalization, call
        :func:`~galpy.potential.SnapshotPotential.InterpSnapshotPotential.denormalize`.

        """

        Vc0 = self.vcirc(R0)
        Phi0 = numpy.abs(self.Rforce(R0, 0.0))

        self._normR0 = R0
        self._normVc0 = Vc0
        self._normPhi0 = Phi0

        # rescale the simulation
        if not isinstance(self._s["pos"].units, NoUnit):
            self._posunit = self._s["pos"].units
            self._s["pos"].convert_units("%s kpc" % R0)
        else:
            self._posunit = None
        if not isinstance(self._s["vel"].units, NoUnit):
            self._velunit = self._s["vel"].units
            self._s["vel"].convert_units("%s km s**-1" % Vc0)
        else:
            self._velunit = None

        # rescale the grid
        self._rgrid /= R0
        if self._logR:
            self._logrgrid -= numpy.log(R0)
            rs = self._logrgrid
        else:
            rs = self._rgrid

        self._zgrid /= R0

        # rescale the potential
        self._amp /= Phi0

        self._savedsplines = {}

        # rescale anything using splines
        if not self._enable_c and self._interpPot:
            for spline, name in zip(
                [self._potInterp, self._rforceInterp, self._zforceInterp],
                ["pot", "rforce", "zforce"],
            ):
                self._savedsplines[name] = spline

            self._potInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._potGrid / R0, kx=3, ky=3, s=0.0
            )
            self._rforceInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._rforceGrid, kx=3, ky=3, s=0.0
            )
            self._zforceInterp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._zforceGrid, kx=3, ky=3, s=0.0
            )
        elif self._enable_c and self._interpPot:
            self._potGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._potGrid / R0)

        if self._interpPot:
            self._savedsplines["vcirc"] = self._vcircInterp
            self._vcircInterp = interpolate.InterpolatedUnivariateSpline(
                rs, self._vcircGrid / Vc0, k=3
            )

        if self._interpepifreq:
            self._savedsplines["R2deriv"] = self._R2interp
            self._savedsplines["epifreq"] = self._epifreqInterp
            self._R2interp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._R2derivGrid, kx=3, ky=3, s=0.0
            )
            self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                rs[self._epigoodindx],
                self._epifreqGrid[self._epigoodindx] / numpy.sqrt(Phi0 / R0),
                k=3,
            )

        if self._interpverticalfreq:
            self._savedsplines["z2deriv"] = self._z2interp
            self._savedsplines["verticalfreq"] = self._verticalfreqInterp
            self._z2interp = interpolate.RectBivariateSpline(
                rs, self._zgrid, self._z2derivGrid, kx=3, ky=3, s=0.0
            )
            self._verticalfreqInterp = interpolate.InterpolatedUnivariateSpline(
                rs[self._verticalgoodindx],
                self._verticalfreqGrid[self._verticalgoodindx] / numpy.sqrt(Phi0 / R0),
                k=3,
            )

    def denormalize(self):
        """

        Undo the normalization.

        """
        R0 = self._normR0
        Vc0 = self._normVc0
        Phi0 = self._normPhi0

        # rescale the simulation
        if not self._posunit is None:
            self._s["pos"].convert_units(self._posunit)
        if not self._velunit is None:
            self._s["vel"].convert_units(self._velunit)

        # rescale the grid
        self._rgrid *= R0
        if self._logR:
            self._logrgrid += numpy.log(R0)
            rs = self._logrgrid
        else:
            rs = self._rgrid

        self._zgrid *= R0

        # rescale the potential
        self._amp *= Phi0

        # restore the splines
        if not self._enable_c and self._interpPot:
            self._potInterp = self._savedsplines["pot"]
            self._rforceInterp = self._savedsplines["rforce"]
            self._zforceInterp = self._savedsplines["zforce"]
        elif self._enable_c and self._interpPot:
            self._potGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._potGrid)

        if self._interpPot:
            self._vcircInterp = self._savedsplines["vcirc"]

        if self._interpepifreq:
            self._R2interp = self._savedsplines["R2deriv"]
            self._epifreqInterp = self._savedsplines["epifreq"]

        if self._interpverticalfreq:
            self._z2interp = self._savedsplines["z2deriv"]
            self._verticalfreqInterp = self._savedsplines["verticalfreq"]

    # Pickling functions
    def __getstate__(self):
        pdict = copy.copy(self.__dict__)
        # Deconstruct _s
        pdict["_pos"] = self._s["pos"]
        pdict["_mass"] = self._s["mass"]
        pdict["_eps"] = self._s["eps"]
        # rm _s and _origPot,
        del pdict["_s"]
        del pdict["_origPot"]
        return pdict

    def __setstate__(self, pdict):
        # Set up snapshot again for origPot
        pdict["_s"] = pynbody.new(star=len(pdict["_mass"]))
        pdict["_s"]["pos"] = pdict["_pos"]
        pdict["_s"]["mass"] = pdict["_mass"]
        pdict["_s"]["eps"] = pdict["_eps"]
        # Transfer __dict__
        del pdict["_pos"]
        del pdict["_mass"]
        del pdict["_eps"]
        self.__dict__ = pdict
        # Now setup origPotnagain
        self._origPot = SnapshotRZPotential(self._s, self._numcores)
        return None
