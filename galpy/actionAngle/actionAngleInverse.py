###############################################################################
# actionAngleInverse.py: top-level class with common routines for inverse
#                        actionAngle methods
###############################################################################
import numpy

from ..util.conversion import (
    actionAngleInverse_physical_input,
    physical_conversion_actionAngleInverse,
)
from .actionAngle import actionAngle


class actionAngleInverse(actionAngle):
    """actionAngleInverse; top-level class with common routines for inverse actionAngle methods"""

    def __init__(self, *args, **kwargs):
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("__call__", pop=True)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) at the given actions and angles.

        The actions and the angles broadcast against each other by numpy's rules: scalar actions with arrays of angles evaluate one torus at those angles; arrays of actions with scalar angles evaluate that angle on every torus; arrays of the same shape are paired element by element, each star on its own torus at its own angle; and actions of shape (n,1) with angles of shape (m,) return every angle on every torus as (n,m) arrays.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            (R,vR,vT,z,vz,phi), each in the broadcast shape of the actions and angles.

        Notes
        -----
        - 2017-11-14 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions, broadcast against the angles - Bovy (UofT)

        """
        try:
            return self._evaluate(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'__call__' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("xvFreqs", pop=True)
    def xvFreqs(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) and the frequencies at the given actions and angles.

        The actions and the angles broadcast against each other by numpy's rules: scalar actions with arrays of angles evaluate one torus at those angles; arrays of actions with scalar angles evaluate that angle on every torus; arrays of the same shape are paired element by element, each star on its own torus at its own angle; and actions of shape (n,1) with angles of shape (m,) return every angle on every torus as (n,m) arrays.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            A tuple containing the phase-space coordinates (R,vR,vT,z,vz,phi) and the frequencies (OmegaR,Omegaphi,Omegaz), each in the broadcast shape of the actions and angles; the frequencies are floats for a single torus.

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions, broadcast against the angles - Bovy (UofT)

        """
        try:
            return self._xvFreqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'xvFreqs' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("Freqs", pop=True)
    def Freqs(self, *args, **kwargs):
        """
        Return the frequencies of a torus or of an array of tori

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.

        Returns
        -------
        tuple
            The frequencies (OmegaR, Omegaphi, Omegaz): three floats for a single torus, three arrays in the broadcast shape of the actions otherwise.

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions - Bovy (UofT)

        """
        try:
            return self._Freqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'Freqs' method not implemented for this actionAngle module"
            )

    def _invert(
        self,
        R,
        vR,
        vT,
        z,
        vz,
        phi,
        jr,
        jz,
        angler,
        anglez,
        fixed=False,
        tol=1e-12,
        maxiter=30,
    ):
        """The actions and angles that this instance maps to the points
        (R, v_R, v_T, z, v_z, phi) (flat arrays, internal units), by
        Newton's method on the map itself from the guess (jr, jz, angler,
        anglez): L_z = R v_T is exact and phi is linear in theta_phi, so
        the unknowns are the two libration planes' sqrt(2 J) (cos theta,
        sin theta) (regular at a vanishing action) and the equations are
        (R, v_R, z, v_z); the Jacobian by central differences, kept while
        the steps contract the residual twentyfold and renewed otherwise;
        a step is halved until it lowers the residual without leaving the
        domain. With fixed, the actions are held at (jr, jz) and the unknowns
        are the two angles alone (the least-squares step of the four
        equations). Returns (J_R, J_z, theta_R, theta_phi, theta_z), nan for
        the points whose residual does not reach tol (outside the domain,
        or with a guess too far)"""
        n, h = len(R), 1e-6
        Lz = R * vT
        x = numpy.array([R, vR, z, vz])
        if fixed:
            q = numpy.array([angler, anglez])
        else:
            q = numpy.sqrt(2.0 * numpy.array([jr, jr, jz, jz])) * numpy.array(
                [
                    numpy.cos(angler),
                    numpy.sin(angler),
                    numpy.cos(anglez),
                    numpy.sin(anglez),
                ]
            )
        nq = len(q)

        def unpack(q, idx):
            if fixed:
                twopi = 2.0 * numpy.pi
                return jr[idx], jz[idx], q[0] % twopi, q[1] % twopi
            return (
                0.5 * (q[0] ** 2 + q[1] ** 2),
                0.5 * (q[2] ** 2 + q[3] ** 2),
                numpy.arctan2(q[1], q[0]) % (2.0 * numpy.pi),
                numpy.arctan2(q[3], q[2]) % (2.0 * numpy.pi),
            )

        def point(q, idx):
            # (R, v_R, z, v_z, phi) at theta_phi = 0; nan outside the domain
            qjr, qjz, qtr, qtz = unpack(q, idx)
            out = numpy.full((5, len(idx)), numpy.nan)
            ok = numpy.all(numpy.isfinite(q), axis=0)
            ok[ok] = self.in_domain(qjr[ok], Lz[idx][ok], qjz[ok], use_physical=False)
            if numpy.any(ok):
                o = self._evaluate(
                    qjr[ok], Lz[idx][ok], qjz[ok], qtr[ok], 0.0 * qtr[ok], qtz[ok]
                )
                out[:, ok] = o[0], o[1], o[3], o[4], o[5]
            return out

        def residual(f, idx):
            return numpy.amax(numpy.fabs(f[:4] - x[:, idx]), axis=0)

        f = point(q, numpy.arange(n))
        err = residual(f, numpy.arange(n))
        jac, rate = numpy.tile(numpy.eye(4)[:, :nq], (n, 1, 1)), numpy.ones(n)
        active = err >= tol
        frozen = numpy.zeros((nq, n), dtype="bool")

        def newton():
            for _ in range(maxiter):
                idx = numpy.flatnonzero(active)
                if len(idx) == 0:
                    break
                new = idx[rate[idx] > 0.05]
                for k in range(nq):
                    dq = numpy.zeros((nq, 1))
                    dq[k] = h
                    jac[new, :, k] = (
                        (
                            point(q[:, new] + dq, new)[:4]
                            - point(q[:, new] - dq, new)[:4]
                        )
                        / (2.0 * h)
                    ).T
                # (a Jacobian that reaches outside the domain: no step)
                lost = ~numpy.all(numpy.isfinite(jac[idx]), axis=(1, 2))
                jac[idx] *= ~frozen[:, idx].T[:, None, :]
                jac[idx[lost]] = numpy.eye(4)[:, :nq]
                step = (
                    numpy.linalg.pinv(jac[idx]) @ (x - f[:4])[:, idx].T[:, :, None]
                )[:, :, 0].T
                step[:, lost] = numpy.nan
                step[frozen[:, idx]] = 0.0
                todo, t = numpy.arange(len(idx)), 1.0
                for _ in range(6):
                    sub = idx[todo]
                    qn = q[:, sub] + t * step[:, todo]
                    fn = point(qn, sub)
                    en = residual(fn, sub)
                    better = en < err[sub]
                    acc = sub[better]
                    rate[acc] = en[better] / err[acc]
                    q[:, acc], f[:, acc], err[acc] = (
                        qn[:, better],
                        fn[:, better],
                        en[better],
                    )
                    todo, t = todo[~better], 0.5 * t
                    if len(todo) == 0:
                        break
                # a step that fails with a kept Jacobian gets a new one; with a
                # new one, the point is given up
                stuck = idx[todo]
                active[stuck[numpy.isin(stuck, new)]] = False
                rate[stuck] = 1.0
                active[idx] &= err[idx] >= tol
            return None

        newton()
        if not fixed:
            # a point without one of the two librations: the family's map is
            # not continuous to rounding between a vanishing action and a
            # small one, so the solve stalls near zero; hold that action at
            # zero and solve for the rest
            snap = numpy.zeros(n, dtype="bool")
            for p in (0, 2):
                flat = (
                    numpy.isfinite(err)
                    & (err >= tol)
                    & (q[p] ** 2 + q[p + 1] ** 2 < 1e-12)
                )
                q[p : p + 2, flat], frozen[p : p + 2, flat] = 0.0, True
                snap |= flat
            idx = numpy.flatnonzero(snap)
            if len(idx) > 0:
                f[:, idx] = point(q[:, idx], idx)
                err[idx] = residual(f[:, idx], idx)
                active[idx], rate[idx] = err[idx] >= tol, 1.0
                newton()
        qjr, qjz, qtr, qtz = unpack(q, numpy.arange(n))
        out = numpy.array([qjr, qjz, qtr, (phi - f[4]) % (2.0 * numpy.pi), qtz])
        out[:, ~(err < tol)] = numpy.nan
        return tuple(out)
