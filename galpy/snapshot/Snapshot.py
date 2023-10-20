import numpy as nu
from directnbody import direct_nbody

from galpy.orbit import Orbit
from galpy.potential.planarPotential import RZToplanarPotential
from galpy.util import plot


class Snapshot:
    """General snapshot = collection of particles class"""

    def __init__(self, *args, **kwargs):
        """
        Initialize a snapshot object.

        Parameters
        ----------
        *args : list
            List of orbits, list of masses (masses=)
        **kwargs : dict
            Coming soon:
            1) observations
            2) DFs to draw from

        Notes
        -----
        - 2011-02-02 - Started - Bovy
        """
        if isinstance(args[0], list) and isinstance(args[0][0], Orbit):
            self.orbits = args[0]
            if kwargs.has_key("masses"):
                self.masses = kwargs["masses"]
            else:
                self.masses = nu.ones(len(self.orbits))
        return None

    def integrate(self, t, pot=None, method="test-particle", **kwargs):
        """
        Integrate the snapshot in time.

        Parameters
        ----------
        t : numpy.ndarray
            Times to save the snapshots at (must start at 0).
        pot : object or list of objects, optional
            Potential object(s) (default=None).
        method : str, optional
            Method to use ('test-particle' or 'direct-python' for now).
        **kwargs
            Additional keyword arguments to pass to the integration method.

        Returns
        -------
        list
            List of snapshots at times t.

        Notes
        -----
        - 2011-02-02 - Written - Bovy (NYU)
        """

        if method.lower() == "test-particle":
            return self._integrate_test_particle(t, pot)
        elif method.lower() == "direct-python":
            return self._integrate_direct_python(t, pot, **kwargs)

    def _integrate_test_particle(self, t, pot):
        """Integrate the snapshot as a set of test particles in an external \
        potential"""
        # Integrate all the orbits
        for o in self.orbits:
            o.integrate(t, pot)
        # Return them as a set of snapshots
        out = []
        for ii in range(len(t)):
            outOrbits = []
            for o in self.orbits:
                outOrbits.append(o(t[ii]))
            out.append(Snapshot(outOrbits, self.masses))
        return out

    def _integrate_direct_python(self, t, pot, **kwargs):
        """Integrate the snapshot using a direct force summation method \
        written entirely in python"""
        # Prepare input for direct_nbody
        q = []
        p = []
        nq = len(self.orbits)
        dim = self.orbits[0].dim()
        if pot is None:
            thispot = None
        elif dim == 2:
            thispot = RZToplanarPotential(pot)
        else:
            thispot = pot
        for ii in range(nq):
            # Transform to rectangular frame
            if dim == 1:
                thisq = nu.array([self.orbits[ii].x()]).flatten()
                thisp = nu.array([self.orbits[ii].vx()]).flatten()
            elif dim == 2:
                thisq = nu.array([self.orbits[ii].x(), self.orbits[ii].y()]).flatten()
                thisp = nu.array([self.orbits[ii].vx(), self.orbits[ii].vy()]).flatten()
            elif dim == 3:
                thisq = nu.array(
                    [self.orbits[ii].x(), self.orbits[ii].y(), self.orbits[ii].z()]
                ).flatten()
                thisp = nu.array(
                    [self.orbits[ii].vx(), self.orbits[ii].vy(), self.orbits[ii].vz()]
                ).flatten()
            q.append(thisq)
            p.append(thisp)
        # Run simulation
        nbody_out = direct_nbody(q, p, self.masses, t, pot=thispot, **kwargs)
        # Post-process output
        nt = len(nbody_out)
        out = []
        for ii in range(nt):
            snap_orbits = []
            for jj in range(nq):
                if dim == 3:
                    # go back to the cylindrical frame
                    R = nu.sqrt(
                        nbody_out[ii][0][jj][0] ** 2.0 + nbody_out[ii][0][jj][1] ** 2.0
                    )
                    phi = nu.arccos(nbody_out[ii][0][jj][0] / R)
                    if nbody_out[ii][0][jj][1] < 0.0:
                        phi = 2.0 * nu.pi - phi
                    vR = nbody_out[ii][1][jj][0] * nu.cos(phi) + nbody_out[ii][1][jj][
                        1
                    ] * nu.sin(phi)
                    vT = nbody_out[ii][1][jj][1] * nu.cos(phi) - nbody_out[ii][1][jj][
                        0
                    ] * nu.sin(phi)
                    vxvv = nu.zeros(dim * 2)
                    vxvv[3] = nbody_out[ii][0][jj][2]
                    vxvv[4] = nbody_out[ii][1][jj][2]
                    vxvv[0] = R
                    vxvv[1] = vR
                    vxvv[2] = vT
                    vxvv[5] = phi
                if dim == 2:
                    # go back to the cylindrical frame
                    R = nu.sqrt(
                        nbody_out[ii][0][jj][0] ** 2.0 + nbody_out[ii][0][jj][1] ** 2.0
                    )
                    phi = nu.arccos(nbody_out[ii][0][jj][0] / R)
                    if nbody_out[ii][0][jj][1] < 0.0:
                        phi = 2.0 * nu.pi - phi
                    vR = nbody_out[ii][1][jj][0] * nu.cos(phi) + nbody_out[ii][1][jj][
                        1
                    ] * nu.sin(phi)
                    vT = nbody_out[ii][1][jj][1] * nu.cos(phi) - nbody_out[ii][1][jj][
                        0
                    ] * nu.sin(phi)
                    vxvv = nu.zeros(dim * 2)
                    vxvv[0] = R
                    vxvv[1] = vR
                    vxvv[2] = vT
                    vxvv[3] = phi
                if dim == 1:
                    vxvv = [nbody_out[ii][0][jj], nbody_out[ii][1][jj]]
                snap_orbits.append(Orbit(vxvv))
            out.append(Snapshot(snap_orbits, self.masses))
        return out

    # Plotting
    def plot(self, *args, **kwargs):
        """
        Plot the snapshot (with reasonable defaults)

        Parameters
        ----------
        d1 : str, optional
            First dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...).
        d2 : str, optional
            Second dimension to plot.
        *args : tuple
            Matplotlib.plot inputs + galpy.util.plot.plot inputs.
        **kwargs : dict
            Matplotlib.plot inputs + galpy.util.plot.plot inputs.

        Returns
        -------
        None
            Sends plot to output device.

        Notes
        -----
        - 2011-02-06 - Written based on Orbit's plot
        """
        labeldict = {
            "t": r"$t$",
            "R": r"$R$",
            "vR": r"$v_R$",
            "vT": r"$v_T$",
            "z": r"$z$",
            "vz": r"$v_z$",
            "phi": r"$\phi$",
            "x": r"$x$",
            "y": r"$y$",
            "vx": r"$v_x$",
            "vy": r"$v_y$",
        }
        # Defaults
        if not kwargs.has_key("d1") and not kwargs.has_key("d2"):
            if len(self.orbits[0].vxvv) == 3:
                d1 = "R"
                d2 = "vR"
            elif len(self.orbits[0].vxvv) == 4:
                d1 = "x"
                d2 = "y"
            elif len(self.orbits[0].vxvv) == 2:
                d1 = "x"
                d2 = "vx"
            elif len(self.orbits[0].vxvv) == 5 or len(self.orbits[0].vxvv) == 6:
                d1 = "R"
                d2 = "z"
        elif not kwargs.has_key("d1"):
            d2 = kwargs["d2"]
            kwargs.pop("d2")
            d1 = "t"
        elif not kwargs.has_key("d2"):
            d1 = kwargs["d1"]
            kwargs.pop("d1")
            d2 = "t"
        else:
            d1 = kwargs["d1"]
            kwargs.pop("d1")
            d2 = kwargs["d2"]
            kwargs.pop("d2")
        # Get x and y
        if d1 == "R":
            x = [o.R() for o in self.orbits]
        elif d1 == "z":
            x = [o.z() for o in self.orbits]
        elif d1 == "vz":
            x = [o.vz() for o in self.orbits]
        elif d1 == "vR":
            x = [o.vR() for o in self.orbits]
        elif d1 == "vT":
            x = [o.vT() for o in self.orbits]
        elif d1 == "x":
            x = [o.x() for o in self.orbits]
        elif d1 == "y":
            x = [o.y() for o in self.orbits]
        elif d1 == "vx":
            x = [o.vx() for o in self.orbits]
        elif d1 == "vy":
            x = [o.vy() for o in self.orbits]
        elif d1 == "phi":
            x = [o.phi() for o in self.orbits]
        if d2 == "R":
            y = [o.R() for o in self.orbits]
        elif d2 == "z":
            y = [o.z() for o in self.orbits]
        elif d2 == "vz":
            y = [o.vz() for o in self.orbits]
        elif d2 == "vR":
            y = [o.vR() for o in self.orbits]
        elif d2 == "vT":
            y = [o.vT() for o in self.orbits]
        elif d2 == "x":
            y = [o.x() for o in self.orbits]
        elif d2 == "y":
            y = [o.y() for o in self.orbits]
        elif d2 == "vx":
            y = [o.vx() for o in self.orbits]
        elif d2 == "vy":
            y = [o.vy() for o in self.orbits]
        elif d2 == "phi":
            y = [o.phi() for o in self.orbits]

        # Plot
        if not kwargs.has_key("xlabel"):
            kwargs["xlabel"] = labeldict[d1]
        if not kwargs.has_key("ylabel"):
            kwargs["ylabel"] = labeldict[d2]
        if len(args) == 0:
            args = (",",)
        plot.plot(x, y, *args, **kwargs)

    def plot3d(self, *args, **kwargs):
        """
        Plot the snapshot in 3D (with reasonable defaults)

        Parameters
        ----------
        d1 : str, optional
            First dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...).
        d2 : str, optional
            Second dimension to plot.
        d3 : str, optional
            Third dimension to plot.
        *args
            Matplotlib.plot inputs+galpy.util.plot.plot3d inputs.
        **kwargs
            Matplotlib.plot inputs+galpy.util.plot.plot3d inputs.

        Returns
        -------
        None
            Sends plot to output device.

        Notes
        -----
        - 2011-02-06 - Written based on Orbit's plot3d

        """
        labeldict = {
            "t": r"$t$",
            "R": r"$R$",
            "vR": r"$v_R$",
            "vT": r"$v_T$",
            "z": r"$z$",
            "vz": r"$v_z$",
            "phi": r"$\phi$",
            "x": r"$x$",
            "y": r"$y$",
            "vx": r"$v_x$",
            "vy": r"$v_y$",
        }
        # Defaults
        if (
            not kwargs.has_key("d1")
            and not kwargs.has_key("d2")
            and not kwargs.has_key("d3")
        ):
            if len(self.orbits[0].vxvv) == 3:
                d1 = "R"
                d2 = "vR"
                d3 = "vT"
            elif len(self.orbits[0].vxvv) == 4:
                d1 = "x"
                d2 = "y"
                d3 = "vR"
            elif len(self.orbits[0].vxvv) == 2:
                raise AttributeError("Cannot plot 3D aspects of 1D orbits")
            elif len(self.orbits[0].vxvv) == 5:
                d1 = "R"
                d2 = "vR"
                d3 = "z"
            elif len(self.orbits[0].vxvv) == 6:
                d1 = "x"
                d2 = "y"
                d3 = "z"
        elif not (
            kwargs.has_key("d1") and kwargs.has_key("d2") and kwargs.has_key("d3")
        ):
            raise AttributeError("Please provide 'd1', 'd2', and 'd3'")
        else:
            d1 = kwargs["d1"]
            kwargs.pop("d1")
            d2 = kwargs["d2"]
            kwargs.pop("d2")
            d3 = kwargs["d3"]
            kwargs.pop("d3")
        # Get x, y, and z
        if d1 == "R":
            x = [o.R() for o in self.orbits]
        elif d1 == "z":
            x = [o.z() for o in self.orbits]
        elif d1 == "vz":
            x = [o.vz() for o in self.orbits]
        elif d1 == "vR":
            x = [o.vR() for o in self.orbits]
        elif d1 == "vT":
            x = [o.vT() for o in self.orbits]
        elif d1 == "x":
            x = [o.x() for o in self.orbits]
        elif d1 == "y":
            x = [o.y() for o in self.orbits]
        elif d1 == "vx":
            x = [o.vx() for o in self.orbits]
        elif d1 == "vy":
            x = [o.vy() for o in self.orbits]
        elif d1 == "phi":
            x = [o.phi() for o in self.orbits]
        if d2 == "R":
            y = [o.R() for o in self.orbits]
        elif d2 == "z":
            y = [o.z() for o in self.orbits]
        elif d2 == "vz":
            y = [o.vz() for o in self.orbits]
        elif d2 == "vR":
            y = [o.vR() for o in self.orbits]
        elif d2 == "vT":
            y = [o.vT() for o in self.orbits]
        elif d2 == "x":
            y = [o.x() for o in self.orbits]
        elif d2 == "y":
            y = [o.y() for o in self.orbits]
        elif d2 == "vx":
            y = [o.vx() for o in self.orbits]
        elif d2 == "vy":
            y = [o.vy() for o in self.orbits]
        elif d2 == "phi":
            y = [o.phi() for o in self.orbits]
        if d3 == "R":
            z = [o.R() for o in self.orbits]
        elif d3 == "z":
            z = [o.z() for o in self.orbits]
        elif d3 == "vz":
            z = [o.vz() for o in self.orbits]
        elif d3 == "vR":
            z = [o.vR() for o in self.orbits]
        elif d3 == "vT":
            z = [o.vT() for o in self.orbits]
        elif d3 == "x":
            z = [o.x() for o in self.orbits]
        elif d3 == "y":
            z = [o.y() for o in self.orbits]
        elif d3 == "vx":
            z = [o.vx() for o in self.orbits]
        elif d3 == "vy":
            z = [o.vy() for o in self.orbits]
        elif d3 == "phi":
            z = [o.phi() for o in self.orbits]

        # Plot
        if not kwargs.has_key("xlabel"):
            kwargs["xlabel"] = labeldict[d1]
        if not kwargs.has_key("ylabel"):
            kwargs["ylabel"] = labeldict[d2]
        if not kwargs.has_key("zlabel"):
            kwargs["zlabel"] = labeldict[d3]
        if len(args) == 0:
            args = (",",)
        plot.plot3d(x, y, z, *args, **kwargs)

    # Pickling
    def __getstate__(self):
        return (self.orbits, self.masses)

    def __setstate__(self, state):
        self.orbits = state[0]
        self.masses = state[1]
