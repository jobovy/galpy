"""Specs for test_backend_param_sweep.py.

A spec is a dict:

* ``name``: unique, no whitespace (it is a column of backend_param_known_gaps.txt);
* ``build(P) -> obj`` and ``outputs(obj, C) -> {name: scalar}``; the object is
  built INSIDE the differentiated function, so constructor parameters are on the
  graph;
* ``params``/``coords``: {name: float}; every param and the ``diff_coords``
  (default: the first coordinate) are differentiated;
* ``fd``: "numpy" (Richardson FD on numpy, default), "self" (FD on the same
  backend; sampled outputs, where numpy's RNG differs from the backend key) or
  "ref:<spec>" (numpy FD of a closed-form twin with the same params/coords);
* ``fdscale``, ``self_fd_h``: FD step controls; ``rtol`` (default 1e-6);
* ``covers``: the public classes the spec exercises (default: the name up to
  the first "[" or "(").
"""

import numpy as np

import galpy.actionAngle as aa
import galpy.df as gdf
import galpy.potential as gp
from galpy.backend import get_namespace as _xp
from galpy.df import jeans
from galpy.orbit import Orbit
from galpy.util import conversion as cv
from galpy.util import coords as gc

SPECS = []


def add(name, build, params, outputs, coords, group, **kw):
    SPECS.append(
        dict(
            name=name,
            build=build,
            params=params,
            outputs=outputs,
            coords=dict(coords),
            group=group,
            **kw,
        )
    )


def bk(*vals):
    for v in vals:
        m = type(v).__module__
        if m.startswith("jax"):
            return "jax"
        if m.startswith("torch"):
            return "torch"
    return None


def vec(*vals):
    b = bk(*vals)
    if b == "jax":
        import jax.numpy as jnp

        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in vals])
    if b == "torch":
        import torch

        return torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in vals])
    return np.array([float(v) for v in vals])


def _forced(b):
    import contextlib

    from galpy.backend import use

    return use(b, force=True) if b else contextlib.nullcontext()


###############################################################################
# galpy.potential
###############################################################################
C3 = {"R": 0.9, "z": 0.2, "phi": 0.4, "t": 0.3}
C2 = {"R": 0.9, "phi": 0.4, "t": 0.3}
C1 = {"x": 0.7, "t": 0.3}


def out3(names=("Phi", "Rforce", "zforce", "phitorque", "dens")):
    def f(p, C):
        a = (C["R"], C["z"])
        kw = dict(phi=C["phi"], t=C["t"])
        m = {
            "Phi": lambda: p(*a, **kw),
            "Rforce": lambda: p.Rforce(*a, **kw),
            "zforce": lambda: p.zforce(*a, **kw),
            "phitorque": lambda: p.phitorque(*a, **kw),
            "dens": lambda: p.dens(*a, **kw),
        }
        return {n: m[n]() for n in names}

    return f


def outdiss(p, C):
    kw = dict(phi=C["phi"], t=C["t"], v=[0.1, 1.0, 0.05])
    return {
        "Rforce": p.Rforce(C["R"], C["z"], **kw),
        "zforce": p.zforce(C["R"], C["z"], **kw),
        "phitorque": p.phitorque(C["R"], C["z"], **kw),
    }


def out2(names=("Phi", "Rforce", "phitorque")):
    def f(p, C):
        kw = dict(phi=C["phi"], t=C["t"])
        m = {
            "Phi": lambda: p(C["R"], **kw),
            "Rforce": lambda: p.Rforce(C["R"], **kw),
            "phitorque": lambda: p.phitorque(C["R"], **kw),
        }
        return {n: m[n]() for n in names}

    return f


def out1(p, C):
    return {"Phi": p(C["x"], t=C["t"]), "force": p.force(C["x"], t=C["t"])}


def add_pot(name, build, params, outputs=None, coords=None, **kw):
    add(name, build, params, outputs or out3(), coords or C3, "potential", **kw)


# amp + every float shape parameter
_SIMPLE = {
    "BurkertPotential": {"a": 2.0},
    "DehnenCoreSphericalPotential": {"a": 1.0},
    "DehnenSphericalPotential": {"a": 1.0, "alpha": 1.5},
    "DoubleExponentialDiskPotential": {"hr": 1 / 3.0, "hz": 1 / 16.0},
    "EinastoPotential": {"h": 2.0, "n": 1.3},
    "ExpTruncNFWPotential": {"a": 1.0, "rc": 2.0},
    "FlattenedPowerPotential": {"alpha": 0.5, "q": 0.9, "core": 0.1},
    "HernquistPotential": {"a": 1.0},
    "IsochronePotential": {"b": 1.0},
    "JaffePotential": {"a": 1.0},
    "KeplerPotential": {},
    "KuzminDiskPotential": {"a": 1.0},
    "KuzminKutuzovStaeckelPotential": {"ac": 5.0, "Delta": 1.0},
    "LogarithmicHaloPotential": {"core": 0.1, "q": 0.9, "b": 0.8},
    "MN3ExponentialDiskPotential": {"hr": 1 / 3.0, "hz": 1 / 16.0},
    "MiyamotoNagaiPotential": {"a": 1.0, "b": 0.1},
    "NFWPotential": {"a": 1.0},
    "NullPotential": {},
    "PerfectEllipsoidPotential": {"a": 5.0, "b": 0.9, "c": 0.8},
    "PlummerPotential": {"b": 0.8},
    "PowerSphericalPotential": {"alpha": 1.0},
    "PowerSphericalPotentialwCutoff": {"alpha": 1.0, "rc": 1.0},
    "PowerTriaxialPotential": {"alpha": 1.0, "b": 0.9, "c": 0.8},
    "PseudoIsothermalPotential": {"a": 1.0},
    "RazorThinExponentialDiskPotential": {"hr": 1 / 3.0},
    "RingPotential": {"a": 0.75},
    "SphericalShellPotential": {"a": 0.75},
    "TriaxialGaussianPotential": {"sigma": 5.0, "b": 0.9, "c": 0.8},
    "TriaxialHernquistPotential": {"a": 2.0, "b": 0.9, "c": 0.8},
    "TriaxialJaffePotential": {"a": 2.0, "b": 0.9, "c": 0.8},
    "TriaxialNFWPotential": {"a": 2.0, "b": 0.9, "c": 0.8},
    "TwoPowerSphericalPotential": {"a": 5.0, "alpha": 1.5, "beta": 3.5},
    "TwoPowerTriaxialPotential": {
        "a": 5.0,
        "alpha": 1.5,
        "beta": 3.5,
        "b": 0.9,
        "c": 0.8,
    },
    "SoftenedNeedleBarPotential": {
        "a": 4.0,
        "b": 0.1,
        "c": 1.0,
        "pa": 0.4,
        "omegab": 1.8,
    },
    "FerrersPotential": {"a": 1.0, "b": 0.35, "c": 0.2375, "omegab": 0.3, "pa": 0.1},
    "DehnenBarPotential": {
        "chi": 0.8,
        "rolr": 0.9,
        "barphi": 0.4,
        "tform": -0.1,
        "tsteady": 2.0,
        "beta": 0.1,
        "alpha": 0.01,
    },
    "SpiralArmsPotential": {
        "alpha": 0.2,
        "r_ref": 1.0,
        "phi_ref": 0.1,
        "Rs": 0.3,
        "H": 0.125,
        "omega": 0.2,
    },
}
for _n, _ps in _SIMPLE.items():
    _cls = getattr(gp, _n)
    _outs = ("Phi", "Rforce", "zforce", "phitorque", "dens")
    if _n == "RazorThinExponentialDiskPotential":
        _outs = ("Phi", "Rforce", "zforce")
    elif _n == "KuzminDiskPotential":
        _outs = ("Phi", "Rforce", "zforce", "phitorque")
    elif _n == "NullPotential":
        _outs = ("Phi", "Rforce", "zforce")
    add_pot(_n, lambda P, cls=_cls: cls(**P), {"amp": 1.3, **_ps}, outputs=out3(_outs))
add_pot(
    "HomogeneousSpherePotential",
    lambda P: gp.HomogeneousSpherePotential(amp=P["amp"], R=P["Rsphere"]),
    {"amp": 1.3, "Rsphere": 1.1},
)
add_pot(
    "HenonHeilesPotential",
    lambda P: gp.HenonHeilesPotential(**P),
    {"amp": 1.0},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "DehnenBarPotential[omegab,rb,Af]",
    lambda P: gp.DehnenBarPotential(
        omegab=P["omegab"], rb=P["rb"], Af=P["Af"], barphi=0.4
    ),
    {"omegab": 1.85, "rb": 0.8, "Af": 0.01},
)
add_pot(
    "NFWPotential[conc,mvir]",
    lambda P: gp.NFWPotential(conc=P["conc"], mvir=P["mvir"], H=P["H"], Om=P["Om"]),
    {"conc": 10.0, "mvir": 1.0, "H": 70.0, "Om": 0.3},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
add_pot(
    "NFWPotential[rmax,vmax]",
    lambda P: gp.NFWPotential(rmax=P["rmax"], vmax=P["vmax"]),
    {"rmax": 2.0, "vmax": 0.8},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
add_pot(
    "EinastoPotential[rs]",
    lambda P: gp.EinastoPotential(amp=P["amp"], rs=P["rs"], n=P["n"]),
    {"amp": 1.0, "rs": 1.5, "n": 2.0},
)
add_pot(
    "ExpTruncNFWPotential[mass]",
    lambda P: gp.ExpTruncNFWPotential(mass=P["mass"], a=P["a"], rc=P["rc"]),
    {"mass": 1.2, "a": 1.0, "rc": 2.0},
)
add_pot(
    "PlummerPotential[normalize]",
    lambda P: gp.PlummerPotential(normalize=P["normalize"], b=P["b"]),
    {"normalize": 0.7, "b": 0.8},
)
add_pot(
    "MiyamotoNagaiPotential[normalize]",
    lambda P: gp.MiyamotoNagaiPotential(normalize=P["normalize"], a=P["a"], b=P["b"]),
    {"normalize": 0.6, "a": 0.5, "b": 0.1},
)
add_pot(
    "DoubleExponentialDiskPotential[normalize]",
    lambda P: gp.DoubleExponentialDiskPotential(
        normalize=P["normalize"], hr=P["hr"], hz=P["hz"]
    ),
    {"normalize": 0.6, "hr": 1 / 3.0, "hz": 1 / 16.0},
    outputs=out3(("Rforce", "zforce")),
)
add_pot(
    "TriaxialNFWPotential[zvec,pa]",
    lambda P: gp.TriaxialNFWPotential(
        amp=1.0, a=2.0, b=0.9, c=0.8, zvec=[np.sin(0.1), 0.0, np.cos(0.1)], pa=P["pa"]
    ),
    {"pa": 0.2},
)
add_pot(
    "AnySphericalPotential",
    lambda P: gp.AnySphericalPotential(
        amp=P["amp"], dens=lambda r: 1.0 / (1 + (r / 1.0) ** 2) ** 2.5
    ),
    {"amp": 1.2},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
add_pot(
    "AnyAxisymmetricRazorThinDiskPotential",
    lambda P: gp.AnyAxisymmetricRazorThinDiskPotential(
        amp=P["amp"], surfdens=lambda R: 1.5 * _xp(R).exp(-3 * R)
    ),
    {"amp": 1.2},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
add_pot(
    "CompositePotential",
    lambda P: (
        gp.MiyamotoNagaiPotential(amp=1.0, a=P["a"], b=0.3)
        + gp.NFWPotential(amp=P["amp"], a=4.0)
    ),
    {"a": 0.5, "amp": 1.6},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)


def _scf(P):
    Acos = np.zeros((3, 2, 1))
    Acos[0, 0, 0] = 1.0
    Acos[1, 0, 0] = 0.1
    Acos[2, 1, 0] = 0.05
    return gp.SCFPotential(amp=P["amp"], Acos=Acos, a=P["a"])


add_pot("SCFPotential", _scf, {"amp": 1.3, "a": 1.2})
add_pot(
    "scf_compute_coeffs_axi[SCFPotential]",
    lambda P: gp.SCFPotential(
        amp=1.0,
        a=P["a"],
        Acos=gp.scf_compute_coeffs_axi(
            lambda R, z: 1.0 / (R**2 + (z / P["q"]) ** 2 + 0.3) ** 2, 4, 2, a=P["a"]
        )[0],
    ),
    {"a": 1.2, "q": 0.8},
    outputs=out3(("Phi", "Rforce", "zforce")),
    covers=("SCFPotential",),
)
add_pot(
    "DiskSCFPotential",
    lambda P: gp.DiskSCFPotential(
        amp=P["amp"],
        dens=lambda R, z: (
            13.5 * _xp(R, z).exp(-3.0 * R) * _xp(R, z).exp(-27.0 * _xp(R, z).abs(z))
        ),
        Sigma={"type": "exp", "h": P["h"], "amp": 1.0},
        hz={"type": "exp", "h": P["hz"]},
        a=1.0,
        N=8,
        L=8,
    ),
    {"amp": 1.0, "h": 1 / 3.0, "hz": 1 / 27.0},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
add_pot(
    "MultipoleExpansionPotential",
    lambda P: gp.MultipoleExpansionPotential.from_density(
        amp=P["amp"],
        dens=lambda R, z, phi: (
            1.0 / (1 + (R**2 + (z / P["q"]) ** 2) / P["a"] ** 2) ** 2
        ),
        rgrid=np.geomspace(1e-3, 30, 301),
        L=6,
    ),
    {"amp": 1.1, "a": 1.0, "q": 0.8},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
add_pot(
    "DiskMultipoleExpansionPotential",
    lambda P: gp.DiskMultipoleExpansionPotential(
        amp=P["amp"],
        Sigma={"type": "exp", "h": P["h"], "amp": 1.0},
        hz={"type": "exp", "h": P["hz"]},
        rgrid=np.geomspace(1e-3, 30, 301),
        L=6,
    ),
    {"amp": 0.9, "h": 1 / 3.0, "hz": 1 / 27.0},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
# dissipative
add_pot(
    "ChandrasekharDynamicalFrictionForce",
    lambda P: gp.ChandrasekharDynamicalFrictionForce(
        amp=P["amp"],
        GMs=P["GMs"],
        gamma=P["gamma"],
        rhm=P["rhm"],
        dens=gp.HernquistPotential(amp=1.0, a=P["a"]),
    ),
    {"amp": 1.0, "GMs": 0.1, "gamma": 1.0, "rhm": 0.05, "a": 1.0},
    outputs=outdiss,
)
add_pot(
    "ChandrasekharDynamicalFrictionForce[sigmar]",
    lambda P: gp.ChandrasekharDynamicalFrictionForce(
        amp=1.0,
        GMs=P["GMs"],
        dens=gp.HernquistPotential(amp=1.0, a=1.0),
        sigmar=lambda r: P["s0"] * np.exp(-r / 5.0),
    ),
    {"GMs": 0.1, "s0": 0.5},
    outputs=outdiss,
)
add_pot(
    "FDMDynamicalFrictionForce",
    lambda P: gp.FDMDynamicalFrictionForce(
        amp=P["amp"],
        GMs=P["GMs"],
        gamma=P["gamma"],
        rhm=P["rhm"],
        m=P["m"],
        dens=gp.HernquistPotential(amp=1.0, a=P["a"]),
    ),
    {"amp": 1.0, "GMs": 0.1, "gamma": 1.0, "rhm": 0.05, "m": 1e-23, "a": 1.0},
    outputs=outdiss,
)
add_pot(
    "NonInertialFrameForce",
    lambda P: gp.NonInertialFrameForce(Omega=P["Omega"], Omegadot=P["Omegadot"]),
    {"Omega": 0.5, "Omegadot": 0.1},
    outputs=outdiss,
)


# wrappers: inner MN params (mna, mnb) + the wrapper's own
def MN(P):
    return gp.MiyamotoNagaiPotential(amp=1.0, a=P["mna"], b=P["mnb"])


W = {"mna": 0.5, "mnb": 0.1}
add_pot(
    "DehnenSmoothWrapperPotential",
    lambda P: gp.DehnenSmoothWrapperPotential(
        amp=P["amp"], pot=MN(P), tform=P["tform"], tsteady=P["tsteady"]
    ),
    {"amp": 1.0, "tform": -0.5, "tsteady": 2.0, **W},
)
add_pot(
    "GaussianAmplitudeWrapperPotential",
    lambda P: gp.GaussianAmplitudeWrapperPotential(
        amp=P["amp"], pot=MN(P), to=P["to"], sigma=P["sigma"]
    ),
    {"amp": 1.0, "to": 0.1, "sigma": 1.0, **W},
)
add_pot(
    "SolidBodyRotationWrapperPotential",
    lambda P: gp.SolidBodyRotationWrapperPotential(
        amp=P["amp"],
        pot=gp.SoftenedNeedleBarPotential(a=P["mna"], omegab=0.0),
        omega=P["omega"],
        pa=P["pa"],
    ),
    {"amp": 1.0, "omega": 1.1, "pa": 0.2, "mna": 2.0},
)
add_pot(
    "CorotatingRotationWrapperPotential",
    lambda P: gp.CorotatingRotationWrapperPotential(
        amp=P["amp"],
        pot=gp.SoftenedNeedleBarPotential(a=P["mna"], omegab=0.0),
        vpo=P["vpo"],
        beta=P["beta"],
        to=P["to"],
        pa=P["pa"],
    ),
    {"amp": 1.0, "vpo": 1.0, "beta": 0.1, "to": 0.0, "pa": 0.2, "mna": 2.0},
)
add_pot(
    "TimeDependentAmplitudeWrapperPotential",
    lambda P: gp.TimeDependentAmplitudeWrapperPotential(
        amp=P["amp"], pot=MN(P), A=lambda t: 1.0 + 0.1 * t
    ),
    {"amp": 1.0, **W},
)
add_pot(
    "RotateAndTiltWrapperPotential[galaxy_pa]",
    lambda P: gp.RotateAndTiltWrapperPotential(
        amp=P["amp"],
        pot=MN(P),
        zvec=[np.sin(0.1), 0.0, np.cos(0.1)],
        galaxy_pa=P["galaxy_pa"],
    ),
    {"amp": 1.0, "galaxy_pa": 0.3, **W},
)
add_pot(
    "RotateAndTiltWrapperPotential[inclination]",
    lambda P: gp.RotateAndTiltWrapperPotential(
        pot=MN(P), inclination=P["inclination"], galaxy_pa=0.3, sky_pa=P["sky_pa"]
    ),
    {"inclination": 0.5, "sky_pa": 0.2, **W},
)
add_pot(
    "RotateAndTiltWrapperPotential[offset]",
    lambda P: gp.RotateAndTiltWrapperPotential(pot=MN(P), offset=[P["ox"], 0.0, 0.1]),
    {"ox": 0.1, **W},
)
add_pot(
    "OblateStaeckelWrapperPotential",
    lambda P: gp.OblateStaeckelWrapperPotential(
        amp=P["amp"], pot=MN(P), delta=P["delta"], u0=P["u0"]
    ),
    {"amp": 1.0, "delta": 0.5, "u0": 1.0, **W},
)
add_pot(
    "CylindricallySeparablePotentialWrapper",
    lambda P: gp.CylindricallySeparablePotentialWrapper(
        amp=P["amp"], pot=MN(P), Rp=P["Rp"]
    ),
    {"amp": 1.0, "Rp": 1.0, **W},
    outputs=out3(("Phi", "Rforce", "zforce", "phitorque")),
)
add_pot(
    "KuzminLikeWrapperPotential",
    lambda P: gp.KuzminLikeWrapperPotential(
        amp=P["amp"], pot=gp.PlummerPotential(b=P["pb"]), a=P["a"], b=P["b"]
    ),
    {"amp": 1.0, "a": 1.1, "b": 0.2, "pb": 0.8},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
add_pot(
    "AdiabaticContractionWrapperPotential",
    lambda P: gp.AdiabaticContractionWrapperPotential(
        amp=P["amp"],
        pot=gp.NFWPotential(amp=2.0, a=P["nfwa"]),
        baryonpot=gp.HernquistPotential(amp=0.5, a=P["ha"]),
        f_bar=P["f_bar"],
        method="cautun",
    ),
    {"amp": 1.0, "nfwa": 2.0, "ha": 0.5, "f_bar": 0.157},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)


def _mo_orbit(P):
    o = Orbit([P["oR"], 0.1, 1.1, 0.1, 0.0, 0.0])
    o.integrate(np.linspace(0.0, 1.0, 101), gp.MWPotential2014, method="dop853_c")
    return o


add_pot(
    "MovingObjectPotential",
    lambda P: gp.MovingObjectPotential(
        _mo_orbit(P), pot=gp.PlummerPotential(amp=0.06, b=P["pb"]), amp=P["amp"]
    ),
    {"amp": 1.0, "pb": 0.1, "oR": 1.0},
    outputs=out3(("Phi", "Rforce", "zforce", "phitorque")),
)
# interpolated / tabulated
add_pot(
    "interpRZPotential",
    lambda P: gp.interpRZPotential(
        RZPot=MN(P),
        rgrid=(np.log(0.01), np.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        interpDens=True,
        zsym=True,
    ),
    dict(W),
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
add_pot(
    "interpSphericalPotential",
    lambda P: gp.interpSphericalPotential(
        rforce=gp.HernquistPotential(amp=P["amp"], a=P["a"]),
        rgrid=np.geomspace(0.01, 5.0, 201),
    ),
    {"amp": 1.0, "a": 1.0},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
add_pot(
    "interpSphericalPotential[callable]",
    lambda P: gp.interpSphericalPotential(
        rforce=lambda r: -P["amp"] / (r + P["a"]) ** 2,
        rgrid=np.geomspace(0.01, 5.0, 201),
        Phi0=-P["amp"] / (0.01 + P["a"]),
    ),
    {"amp": 1.0, "a": 1.0},
    outputs=out3(("Phi", "Rforce", "zforce")),
)
add_pot(
    "KingPotential",
    lambda P: gp.KingPotential(W0=P["W0"], M=P["M"], rt=P["rt"]),
    {"W0": 3.0, "M": 2.3, "rt": 1.4},
    outputs=out3(("Phi", "Rforce", "zforce", "dens")),
)
# planar
add_pot(
    "CosmphiDiskPotential",
    lambda P: gp.CosmphiDiskPotential(
        amp=P["amp"], phib=P["phib"], p=P["p"], phio=P["phio"], m=4, r1=P["r1"]
    ),
    {"amp": 1.0, "phib": 0.4, "p": 1.0, "phio": 0.01, "r1": 1.0},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "CosmphiDiskPotential[rb]",
    lambda P: gp.CosmphiDiskPotential(
        amp=1.0, phib=0.4, p=P["p"], phio=0.01, m=4, rb=P["rb"]
    ),
    {"p": 1.0, "rb": 0.5},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "EllipticalDiskPotential",
    lambda P: gp.EllipticalDiskPotential(
        amp=P["amp"],
        phib=P["phib"],
        p=P["p"],
        twophio=P["twophio"],
        tform=P["tform"],
        tsteady=P["tsteady"],
    ),
    {"amp": 1.0, "phib": 0.4, "p": 1.0, "twophio": 0.01, "tform": -0.5, "tsteady": 2.0},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "LopsidedDiskPotential",
    lambda P: gp.LopsidedDiskPotential(
        amp=P["amp"], phib=P["phib"], p=P["p"], phio=P["phio"]
    ),
    {"amp": 1.0, "phib": 0.4, "p": 1.0, "phio": 0.01},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "SteadyLogSpiralPotential",
    lambda P: gp.SteadyLogSpiralPotential(
        amp=P["amp"],
        omegas=P["omegas"],
        A=P["A"],
        alpha=P["alpha"],
        gamma=P["gamma"],
        tform=P["tform"],
        tsteady=P["tsteady"],
    ),
    {
        "amp": 1.0,
        "omegas": 0.65,
        "A": -0.035,
        "alpha": -7.0,
        "gamma": 0.78,
        "tform": -0.5,
        "tsteady": 2.0,
    },
    outputs=out2(),
    coords=C2,
)
add_pot(
    "TransientLogSpiralPotential",
    lambda P: gp.TransientLogSpiralPotential(
        amp=P["amp"],
        omegas=P["omegas"],
        A=P["A"],
        alpha=P["alpha"],
        gamma=P["gamma"],
        sigma=P["sigma"],
        to=P["to"],
    ),
    {
        "amp": 1.0,
        "omegas": 0.65,
        "A": -0.035,
        "alpha": -7.0,
        "gamma": 0.78,
        "sigma": 1.0,
        "to": 0.1,
    },
    outputs=out2(),
    coords=C2,
)
add_pot(
    "RZToplanarPotential[MN]",
    lambda P: gp.RZToplanarPotential(MN(P)),
    dict(W),
    outputs=out2(("Phi", "Rforce")),
    coords=C2,
    covers=("planarAxiPotential",),
)
add_pot(
    "toPlanarPotential[DehnenBar]",
    lambda P: gp.toPlanarPotential(
        gp.DehnenBarPotential(omegab=P["omegab"], rb=P["rb"], Af=0.01, barphi=0.4)
    ),
    {"omegab": 1.85, "rb": 0.8},
    outputs=out2(),
    coords=C2,
)
add_pot(
    "toPlanarPotential[Composite]",
    lambda P: gp.toPlanarPotential(
        gp.MiyamotoNagaiPotential(amp=1.0, a=P["a"], b=0.3)
        + gp.NFWPotential(amp=P["amp"], a=4.0)
    ),
    {"a": 0.5, "amp": 1.6},
    outputs=out2(("Phi", "Rforce")),
    coords=C2,
    covers=("planarCompositePotential",),
)
add_pot(
    "SolidBodyRotationWrapperPotential[planar]",
    lambda P: gp.SolidBodyRotationWrapperPotential(
        pot=gp.EllipticalDiskPotential(twophio=P["twophio"]), omega=P["omega"]
    ),
    {"twophio": 0.05, "omega": 1.1},
    outputs=out2(),
    coords=C2,
)
# linear
add_pot(
    "KGPotential",
    lambda P: gp.KGPotential(amp=P["amp"], K=P["K"], F=P["F"], D=P["D"]),
    {"amp": 1.0, "K": 1.15, "F": 0.03, "D": 1.8},
    outputs=out1,
    coords=C1,
)
add_pot(
    "IsothermalDiskPotential",
    lambda P: gp.IsothermalDiskPotential(amp=P["amp"], sigma=P["sigma"]),
    {"amp": 1.0, "sigma": 0.1},
    outputs=out1,
    coords=C1,
)
add_pot(
    "RZToverticalPotential[MN]",
    lambda P: gp.RZToverticalPotential(MN(P), P["R"]),
    {**W, "R": 0.9},
    outputs=out1,
    coords=C1,
    covers=("linearPotential",),
)
add_pot(
    "toVerticalPotential[MN]",
    lambda P: gp.toVerticalPotential(MN(P), P["R"]),
    {**W, "R": 0.9},
    outputs=out1,
    coords=C1,
)
add_pot(
    "linearCompositePotential",
    lambda P: (
        gp.KGPotential(amp=P["amp"], K=1.15, F=0.03, D=1.8)
        + gp.IsothermalDiskPotential(amp=1.0, sigma=P["sigma"])
    ),
    {"amp": 1.0, "sigma": 0.1},
    outputs=out1,
    coords=C1,
)


###############################################################################
# galpy.df
###############################################################################
def add_df(name, build, params, outputs, coords, **kw):
    add(name, build, params, outputs, coords, "df", **kw)


PSC = {"R": 0.9, "vR": 0.1, "vT": 0.3, "z": 0.2, "vz": 0.05, "r": 1.1}


def sph_out(moments=True, sig=True):
    def f(df, C):
        o = {"f": df(C["R"], C["vR"], C["vT"], C["z"], C["vz"])}
        if moments:
            o["vm20"] = df.vmomentdensity(C["r"], 2, 0)
        if sig:
            o["sigmar"] = df.sigmar(C["r"])
        return o

    return f


add_df(
    "isotropicHernquistdf",
    lambda P: gdf.isotropicHernquistdf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"])
    ),
    {"amp": 2.3, "a": 1.3},
    sph_out(),
    PSC,
)
add_df(
    "isotropicPlummerdf",
    lambda P: gdf.isotropicPlummerdf(pot=gp.PlummerPotential(amp=P["amp"], b=P["b"])),
    {"amp": 1.7, "b": 1.3},
    sph_out(),
    PSC,
)
add_df(
    "isotropicNFWdf",
    lambda P: gdf.isotropicNFWdf(pot=gp.NFWPotential(amp=P["amp"], a=P["a"]), rmax=1e4),
    {"amp": 2.0, "a": 1.5},
    sph_out(),
    PSC,
)
add_df(
    "isotropicNFWdf[widrow]",
    lambda P: gdf.isotropicNFWdf(
        pot=gp.NFWPotential(amp=P["amp"], a=P["a"]), widrow=True
    ),
    {"amp": 2.0, "a": 1.5},
    sph_out(moments=False, sig=False),
    PSC,
)
add_df(
    "isotropicPowerLawdf",
    lambda P: gdf.isotropicPowerLawdf(
        pot=gp.PowerSphericalPotential(amp=P["amp"], alpha=P["alpha"]),
        rmax=1e4,
        rmin=1e-6,
    ),
    {"amp": 1.0, "alpha": 2.5},
    sph_out(),
    PSC,
)
add_df(
    "osipkovmerrittHernquistdf",
    lambda P: gdf.osipkovmerrittHernquistdf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"]), ra=P["ra"]
    ),
    {"amp": 2.3, "a": 1.3, "ra": 2.3},
    sph_out(),
    PSC,
)
add_df(
    "osipkovmerrittNFWdf",
    lambda P: gdf.osipkovmerrittNFWdf(
        pot=gp.NFWPotential(amp=P["amp"], a=P["a"]), ra=P["ra"]
    ),
    {"amp": 2.3, "a": 1.3, "ra": 2.3},
    sph_out(),
    PSC,
)
add_df(
    "osipkovmerrittPowerLawdf",
    lambda P: gdf.osipkovmerrittPowerLawdf(
        pot=gp.PowerSphericalPotential(amp=P["amp"], alpha=P["alpha"]),
        denspot=gp.PowerSphericalPotential(amp=1.0, alpha=P["dalpha"]),
        ra=P["ra"],
        rmax=100.0,
        rmin=1e-4,
    ),
    {"amp": 1.0, "alpha": 2.5, "dalpha": 2.8, "ra": 2.0},
    sph_out(),
    PSC,
)
add_df(
    "constantbetaHernquistdf",
    lambda P: gdf.constantbetaHernquistdf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"]), beta=P["beta"]
    ),
    {"amp": 2.3, "a": 1.3, "beta": 0.2},
    sph_out(),
    PSC,
)
add_df(
    "constantbetaPowerLawdf",
    lambda P: gdf.constantbetaPowerLawdf(
        pot=gp.PowerSphericalPotential(amp=P["amp"], alpha=P["alpha"]),
        beta=P["beta"],
        rmax=1e4,
        rmin=1e-4,
    ),
    {"amp": 1.0, "alpha": 2.5, "beta": 0.2},
    sph_out(),
    PSC,
)
add_df(
    "eddingtondf",
    lambda P: gdf.eddingtondf(
        pot=gp.DehnenCoreSphericalPotential(amp=P["amp"], a=P["a"]), rmin=1e-4
    ),
    {"amp": 2.5, "a": 1.15},
    sph_out(),
    PSC,
)
add_df(
    "eddingtondf[denspot]",
    lambda P: gdf.eddingtondf(
        pot=gp.NFWPotential(amp=P["amp"], a=P["a"]),
        denspot=gp.DehnenCoreSphericalPotential(amp=0.1, a=P["da"]),
        rmin=1e-4,
    ),
    {"amp": 2.3, "a": 1.3, "da": 0.5},
    sph_out(moments=False),
    PSC,
)
# numerical DFs: FD of the closed-form twin
add_df(
    "eddingtondf[Hernquist]",
    lambda P: gdf.eddingtondf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"]), rmin=1e-4
    ),
    {"amp": 2.3, "a": 1.3},
    sph_out(),
    PSC,
    fd="ref:isotropicHernquistdf",
)
add_df(
    "osipkovmerrittdf",
    lambda P: gdf.osipkovmerrittdf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"]), ra=P["ra"], rmin=1e-4
    ),
    {"amp": 2.3, "a": 1.3, "ra": 2.3},
    sph_out(),
    PSC,
    fd="ref:osipkovmerrittHernquistdf",
    rtol=5e-5,  # numerical OM df differs from the closed form by ~1e-5 (numpy too)
)
add_df(
    "constantbetadf",
    lambda P: gdf.constantbetadf(
        pot=gp.HernquistPotential(amp=P["amp"], a=P["a"]), beta=P["beta"], rmin=1e-4
    ),
    {"amp": 2.3, "a": 1.3, "beta": 0.2},
    sph_out(moments=False),
    PSC,
    fd="ref:constantbetaHernquistdf",
)
add_df(
    "kingdf",
    lambda P: gdf.kingdf(W0=P["W0"], M=P["M"], rt=P["rt"]),
    {"W0": 3.0, "M": 2.3, "rt": 1.76},
    sph_out(),
    {"R": 0.3, "vR": 0.1, "vT": 0.15, "z": 0.1, "vz": 0.05, "r": 0.5},
)
# jeans
_HJ = {"amp": 1.0, "a": 1.3, "beta": 0.2}
add_df(
    "jeans.sigmar",
    lambda P: P,
    dict(_HJ),
    lambda P, C: {
        "sigmar": jeans.sigmar(
            gp.HernquistPotential(amp=P["amp"], a=P["a"]),
            C["r"],
            beta=P["beta"],
            use_physical=False,
        )
    },
    {"r": 1.1},
    covers=(),
)
add_df(
    "jeans.sigmalos",
    lambda P: P,
    dict(_HJ),
    lambda P, C: {
        "sigmalos": jeans.sigmalos(
            gp.HernquistPotential(amp=P["amp"], a=P["a"]),
            C["R"],
            beta=P["beta"],
            use_physical=False,
        )
    },
    {"R": 1.1},
    covers=(),
)
add_df(
    "jeans.sigmalos[dens,surfdens]",
    lambda P: P,
    {"amp": 1.0, "a": 1.3, "da": 0.8},
    lambda P, C: {
        "sigmalos": jeans.sigmalos(
            gp.HernquistPotential(amp=P["amp"], a=P["a"]),
            C["R"],
            dens=lambda r: 1.0 / (1.0 + (r / P["da"]) ** 2) ** 2.5,
            use_physical=False,
        )
    },
    {"R": 1.1},
    covers=(),
)


# disk DFs
def disk_out(df, C):
    from galpy.df.diskdf import vRvTRToEL

    E, L = vRvTRToEL(C["vR"], C["vT"], C["R"], df._beta, df._dftype)
    return {
        "f": df(E, L),
        "surfacemass": df.surfacemass(C["R"]),
        "sigmaR2": df.sigmaR2(C["R"]),
        "meanvT": df.meanvT(C["R"]),
    }


_DC = {"R": 0.9, "vR": 0.1, "vT": 0.9}
for _cls in ("dehnendf", "shudf", "schwarzschilddf"):
    add_df(
        _cls,
        lambda P, cls=_cls: getattr(gdf, cls)(
            beta=P["beta"], profileParams=(P["hR"], P["sR"], P["sigma0"])
        ),
        {"beta": 0.1, "hR": 0.25, "sR": 1.0, "sigma0": 0.2},
        disk_out,
        _DC,
        covers=(_cls, "expSurfaceSigmaProfile", "surfaceSigmaProfile", "diskdf"),
    )
add_df(
    "dehnendf[correct]",
    lambda P: gdf.dehnendf(
        beta=0.0,
        profileParams=(P["hR"], 1.0, P["sigma0"]),
        correct=True,
        niter=2,
        npoints=21,
    ),
    {"hR": 0.25, "sigma0": 0.2},
    disk_out,
    _DC,
    fd="self",
    covers=("dehnendf", "DFcorrection"),
)


def edf_out(edf, C):
    kw = dict(phi=0.2, grid=True, gridpoints=5, integrate_method="dop853_c")
    return {
        "meanvR": edf.meanvR(C["R"], use_physical=False, **kw),
        "sigmaR2": edf.sigmaR2(C["R"], use_physical=False, **kw),
    }


add_df(
    "evolveddiskdf",
    lambda P: gdf.evolveddiskdf(
        gdf.dehnendf(beta=0.0, profileParams=(P["hR"], 1.0, 0.2)),
        pot=gp.LogarithmicHaloPotential(amp=P["lamp"])
        + gp.SteadyLogSpiralPotential(A=P["A"], omegas=P["omegas"]),
        to=-10.0,
    ),
    {"hR": 0.33, "lamp": 1.0, "A": -0.005, "omegas": 0.2},
    edf_out,
    {"R": 0.9},
)

# quasiisothermaldf
_QC = {"R": 0.9, "vR": 0.1, "vT": 0.9, "z": 0.05, "vz": 0.02}


def q_out(q, C):
    cst = (lambda x: x) if bk(C["R"]) else np.asarray
    Rm, zm = cst(C["R"] * 0 + 1.1), cst(C["R"] * 0 + 0.1)
    return {
        "f": q(C["R"], C["vR"], C["vT"], C["z"], C["vz"]),
        "sigmaR2": q.sigmaR2(Rm, zm, gl=True),
        "meanvT": q.meanvT(Rm, zm, gl=True),
    }


def _qiso(P):
    pot = gp.IsochronePotential(amp=P["amp"], b=P["b"])
    return gdf.quasiisothermaldf(
        P["hr"],
        P["sr"],
        P["sz"],
        P["hsr"],
        P["hsz"],
        pot=pot,
        aA=aa.actionAngleIsochrone(ip=pot),
        cutcounter=True,
    )


add_df(
    "quasiisothermaldf[Isochrone]",
    _qiso,
    {"amp": 1.0, "b": 0.6, "hr": 1 / 3.0, "sr": 0.2, "sz": 0.1, "hsr": 1.0, "hsz": 1.0},
    q_out,
    _QC,
)


def _qad(P):
    pot = gp.MiyamotoNagaiPotential(normalize=1.0, a=P["a"], b=0.3)
    return gdf.quasiisothermaldf(
        1 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot,
        aA=aa.actionAngleAdiabatic(pot=pot, c=False),
        cutcounter=True,
        _precomputerg=False,
    )


add_df("quasiisothermaldf[MN,Adiabatic,c=False]", _qad, {"a": 0.5}, q_out, _QC)


def _qst(P):
    pot = [
        gp.MiyamotoNagaiPotential(amp=0.6, a=P["a"], b=0.3),
        gp.NFWPotential(amp=0.4 * 4.0, a=4.0),
    ]
    return gdf.quasiisothermaldf(
        1 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot,
        aA=aa.actionAngleStaeckel(pot=pot, delta=P["delta"], c=True),
        cutcounter=True,
    )


add_df(
    "quasiisothermaldf[Staeckel,c=True]", _qst, {"a": 0.5, "delta": 0.45}, q_out, _QC
)

# impulse kicks (functions; covers=())
_IV = {"vx": 0.5, "vy": 0.1, "vz": 0.2}


def _imp(fn):
    def f(P, C):
        v = vec(C["vx"], C["vy"], C["vz"])
        x = vec(C["vx"] * 0 + 1.2, 0.0, 0.0)
        w = vec(C["vx"] * 0 + 0.1, P["wy"], 0.1)
        k = fn(P, v, x, w)
        return {"k0": k.reshape(-1)[0], "k1": k.reshape(-1)[1]}

    return f


_IMP = {"b": 0.08, "wy": 0.785, "GM": 0.003, "rs": 0.078}
for _fn in ("impulse_deltav_plummer", "impulse_deltav_hernquist"):
    add_df(
        _fn,
        lambda P: P,
        dict(_IMP),
        _imp(
            lambda P, v, x, w, fn=_fn: getattr(gdf, fn)(
                v, x[0:1], P["b"], w, P["GM"], P["rs"]
            )
        ),
        _IV,
        covers=(),
    )
for _fn in (
    "impulse_deltav_plummer_curvedstream",
    "impulse_deltav_hernquist_curvedstream",
):
    add_df(
        _fn,
        lambda P: P,
        dict(_IMP),
        _imp(
            lambda P, v, x, w, fn=_fn: getattr(gdf, fn)(
                v, x, P["b"], w, x, v, P["GM"], P["rs"]
            )
        ),
        _IV,
        covers=(),
    )
_IMPG = {"b": 0.08, "wy": 0.785, "amp": 0.003, "pb": 0.078}
add_df(
    "impulse_deltav_general",
    lambda P: P,
    dict(_IMPG),
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_general(
            v, x[0:1], P["b"], w, gp.PlummerPotential(amp=P["amp"], b=P["pb"])
        )
    ),
    _IV,
    covers=(),
)
add_df(
    "impulse_deltav_general_curvedstream",
    lambda P: P,
    dict(_IMPG),
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_general_curvedstream(
            v, x, P["b"], w, x, v, gp.PlummerPotential(amp=P["amp"], b=P["pb"])
        )
    ),
    _IV,
    covers=(),
)
add_df(
    "impulse_deltav_general_orbitintegration",
    lambda P: P,
    {"b": 0.3, "wy": 0.785, "amp": 0.5, "pb": 0.3, "lamp": 1.0},
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_general_orbitintegration(
            v,
            x,
            P["b"],
            w,
            x,
            v,
            gp.PlummerPotential(amp=P["amp"], b=P["pb"]),
            5.0 * np.pi,
            gp.LogarithmicHaloPotential(amp=P["lamp"]),
            nsamp=200,
        )
    ),
    _IV,
    covers=(),
)
add_df(
    "impulse_deltav_general_fullplummerintegration",
    lambda P: P,
    {"b": 0.3, "wy": 0.785, "GM": 0.5, "rs": 0.3, "lamp": 1.0},
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_general_fullplummerintegration(
            v,
            x,
            P["b"],
            w,
            x,
            v,
            gp.LogarithmicHaloPotential(amp=P["lamp"]),
            P["GM"],
            P["rs"],
            N=200,
        )
    ),
    _IV,
    covers=(),
)
add_df(
    "impulse_deltav_plummerstream",
    lambda P: P,
    {"b": 0.08, "wy": 0.785, "GS": 0.3, "rs": 0.078},
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_plummerstream(
            v, x[0:1], P["b"], w, lambda t: P["GS"], P["rs"], -0.01, 0.01
        )
    ),
    _IV,
    covers=(),
)
add_df(
    "impulse_deltav_plummerstream_curvedstream",
    lambda P: P,
    {"b": 0.08, "wy": 0.785, "GS": 0.3, "rs": 0.078, "lamp": 1.0},
    _imp(
        lambda P, v, x, w: gdf.impulse_deltav_plummerstream_curvedstream(
            v.reshape(1, 3),
            x.reshape(1, 3),
            vec(0.0),
            P["b"],
            w,
            x,
            v,
            lambda t: P["GS"],
            P["rs"],
            gp.LogarithmicHaloPotential(amp=P["lamp"]),
            -0.01,
            0.01,
        )
    ),
    _IV,
    covers=(),
)

# spray DFs: sampled, so FD on the same backend with the same key
_SIC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
_SPRAY = {"mass_e5": 3.0, "lamp": 1.1, "q": 0.9, "pR": 1.2}


def _spray_key(spdf):
    from galpy.backend import random as grandom

    b = spdf._spray_bk
    if b is None:
        np.random.seed(20260707)
        return {}
    return dict(key=grandom.key(20260707, backend=b))


def spray_out(spdf, C):
    s = spdf.sample(n=6, return_orbit=False, tail="leading", **_spray_key(spdf))
    s = s[0] if isinstance(s, (list, tuple)) else s
    return {"sumR": s[0].sum(), "sumvT": s[2].sum()}


def _spray(cls):
    def f(P):
        sp = getattr(gdf, cls)(
            P["mass_e5"] * 1e-5,
            progenitor=Orbit(vec(P["pR"], *_SIC[1:])),
            pot=gp.LogarithmicHaloPotential(amp=P["lamp"], q=P["q"]),
            tdisrupt=1.0,
        )
        sp._spray_bk = bk(*P.values())
        return sp

    return f


for _cls in ("fardal15spraydf", "chen24spraydf", "streamspraydf"):
    add_df(
        _cls,
        _spray(_cls),
        dict(_SPRAY),
        spray_out,
        {"dummy": 0.0},
        fd="self",
        diff_coords=[],
        self_fd_h=(1e-3, 5e-4),
    )


def track_out(spdf, C):
    tr = spdf.streamTrack(n=200, ntp=21, tail="leading", **_spray_key(spdf))
    return {"x": tr.x(C["tp"]), "vy": tr.vy(C["tp"])}


def pair_out(spdf, C):
    tr = spdf.streamTrack(n=200, ntp=21, tail="both", **_spray_key(spdf))
    return {"x": tr.leading.x(C["tp"]), "y": tr.trailing.y(-C["tp"])}


add_df(
    "streamspraydf.streamTrack",
    _spray("chen24spraydf"),
    dict(_SPRAY),
    track_out,
    {"tp": 0.05},
    fd="self",
    self_fd_h=(1e-3, 5e-4),
    covers=("StreamTrack",),
)
add_df(
    "streamspraydf.streamTrack[pair]",
    lambda P: _spray("chen24spraydf")({**_SPRAY, **P}),
    {"lamp": 1.1},
    pair_out,
    {"tp": 0.05},
    fd="self",
    diff_coords=[],
    self_fd_h=(1e-3, 5e-4),
    covers=("StreamTrackPair",),
)

# streamdf / streamgapdf
_STREAM_IC = [
    1.56148083,
    0.35081535,
    -1.15481504,
    0.88719443,
    -0.47713334,
    0.12019596,
]


def _sdf(integrate_method):
    def f(P):
        b = bk(*P.values())
        z0 = sum(P.values()) * 0.0
        lp = gp.LogarithmicHaloPotential(amp=P["lamp"], q=P["q"])
        kw = (
            dict(integrate_method="diffrax")
            if (integrate_method == "inbackend" and b == "jax")
            else {}
        )
        aA = aa.actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=30.0, **kw)
        with _forced(b):
            return gdf.streamdf(
                P["sigv_kms"] / 220.0,
                progenitor=Orbit(vec(P["pR"] + z0, *[z0 + c for c in _STREAM_IC[1:]])),
                pot=lp,
                aA=aA,
                leading=True,
                nTrackChunks=5,
                tdisrupt=4.5 / cv.time_in_Gyr(220.0, 8.0),
                nTrackIterations=0,
            )

    return f


_SDF = {"sigv_kms": 0.365, "lamp": 1.0, "q": 0.9, "pR": _STREAM_IC[0]}
add_df(
    "streamdf",
    _sdf("inbackend"),
    dict(_SDF),
    lambda sdf, C: {
        "density_par": sdf.density_par(C["dangle"]),
        "meanOmega": sdf.meanOmega(C["dangle"], oned=True),
        "sigangledAngle": sdf.sigangledAngle(C["dangle"]),
    },
    {"dangle": 0.3},
)
add_df(
    "streamdf[Cint]",
    _sdf("default"),
    dict(_SDF),
    lambda sdf, C: {
        "density_par": sdf.density_par(C["dangle"]),
        "meanOmega": sdf.meanOmega(C["dangle"], oned=True),
    },
    {"dangle": 0.3},
    fd="ref:streamdf",
    diff_coords=[],
)

_V0, _R0 = 220.0, 8.0
_FULL_IC = [
    2.6556151742081835,
    0.2183747276300308,
    0.67876510797240575,
    -2.0143395648974671,
    -0.3273737682604374,
    0.24218273922966019,
]


def _gap(P):
    b = bk(*P.values())
    z0 = sum(P.values()) * 0.0
    lp = gp.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    kw = dict(integrate_method="diffrax") if b == "jax" else {}
    aA = aa.actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=20.0, ntintJ=1000, **kw)
    with _forced(b):
        return gdf.streamgapdf(
            0.365 * (10.0 / 2.0) ** (1.0 / 3.0) / _V0,
            progenitor=Orbit(vec(*[z0 + c for c in _FULL_IC])),
            pot=lp,
            aA=aA,
            leading=False,
            nTrackChunks=5,
            nTrackIterations=1,
            nTrackChunksImpact=5,
            sigMeanOffset=4.5,
            tdisrupt=10.88 / cv.time_in_Gyr(_V0, _R0),
            impactb=P["impactb"],
            subhalovel=vec(6.82200571, 132.7700529, 149.4174464) / _V0,
            timpact=P["timpact"],
            impact_angle=-2.34,
            GM=P["GM_e3"] * 1e-3,
            rs=P["rs"],
        )


add_df(
    "streamgapdf",
    _gap,
    {
        "GM_e3": 1e3 * 10.0**-2.0 / cv.mass_in_1010msol(_V0, _R0),
        "rs": 0.625 / _R0,
        "impactb": 0.1 / _R0,
        "timpact": 0.88 / cv.time_in_Gyr(_V0, _R0),
    },
    lambda s, C: {
        "density_par": s.density_par(C["dangle"]),
        "meanOmega": s.meanOmega(C["dangle"], oned=True, use_physical=False),
    },
    {"dangle": 0.3},
    diff_coords=[],
)

###############################################################################
# galpy.actionAngle, Orbit, util
###############################################################################
PS = {"R": 1.0, "vR": 0.1, "vT": 1.1, "z": 0.1, "vz": 0.05, "phi": 0.3}
P1 = {"x": 0.1, "vx": 0.05}


def add_aa(name, build, params, outputs, coords=None, group="aa", **kw):
    add(name, build, params, outputs, PS if coords is None else coords, group, **kw)


_AFA = ("jr", "lz", "jz", "Or", "Op", "Oz", "ar", "ap", "az")


def afa(names=_AFA):
    def f(o, C):
        r = o.actionsFreqsAngles(*[C[k] for k in ("R", "vR", "vT", "z", "vz", "phi")])
        return {n: r[i] for i, n in enumerate(_AFA) if n in names}

    return f


def act(o, C):
    r = o(*[C[k] for k in ("R", "vR", "vT", "z", "vz", "phi")])
    return {"jr": r[0], "lz": r[1], "jz": r[2]}


def afa1(o, C):
    r = o.actionsFreqsAngles(C["x"], C["vx"])
    return {"J": r[0], "O": r[1], "a": r[2]}


def MNa(P):
    return gp.MiyamotoNagaiPotential(amp=2.1, a=P["a"], b=P["b"])


MNp = {"a": 0.5, "b": 0.3}
for _c in (False, True):
    add_aa(
        f"actionAngleStaeckel[c={_c}]",
        lambda P, c=_c: aa.actionAngleStaeckel(pot=MNa(P), delta=P["delta"], c=c),
        {**MNp, "delta": 0.4},
        afa(),
        diff_coords=["R", "vz"],
    )
    add_aa(
        f"actionAngleAdiabatic[c={_c}]",
        lambda P, c=_c: aa.actionAngleAdiabatic(pot=MNa(P), gamma=P["gamma"], c=c),
        {**MNp, "gamma": 1.0},
        afa(),
        diff_coords=["R", "vz"],
    )
add_aa(
    "actionAngleIsochrone",
    lambda P: aa.actionAngleIsochrone(b=P["b"]),
    {"b": 1.2},
    afa(),
    diff_coords=["R", "vz"],
)
add_aa(
    "actionAngleIsochrone[ip]",
    lambda P: aa.actionAngleIsochrone(ip=gp.IsochronePotential(amp=P["amp"], b=P["b"])),
    {"amp": 3.0, "b": 1.2},
    afa(),
)
add_aa(
    "actionAngleSpherical",
    lambda P: aa.actionAngleSpherical(pot=gp.HernquistPotential(amp=4.0, a=P["a"])),
    {"a": 0.8},
    afa(),
    diff_coords=["R", "vz"],
)
add_aa(
    "actionAngleSpherical[LogHalo]",
    lambda P: aa.actionAngleSpherical(
        pot=gp.LogarithmicHaloPotential(amp=1.0, core=P["core"])
    ),
    {"core": 0.2},
    afa(),
)
add_aa(
    "actionAngleHarmonic",
    lambda P: aa.actionAngleHarmonic(omega=P["omega"]),
    {"omega": 1.3},
    afa1,
    coords=P1,
)
add_aa(
    "actionAngleVertical",
    lambda P: aa.actionAngleVertical(pot=gp.toVerticalPotential(MNa(P), 1.0)),
    dict(MNp),
    afa1,
    coords=P1,
)
add_aa(
    "actionAngleVertical[IsothermalDisk]",
    lambda P: aa.actionAngleVertical(
        pot=gp.IsothermalDiskPotential(amp=1.0, sigma=P["sigma"])
    ),
    {"sigma": 0.5},
    afa1,
    coords=P1,
)
add_aa(
    "actionAngleIsochroneApprox",
    lambda P: aa.actionAngleIsochroneApprox(
        pot=MNa(P), b=P["bI"], tintJ=20.0, ntintJ=2000
    ),
    {**MNp, "bI": 0.8},
    afa(("jr", "jz", "Or", "ar")),
)
add_aa(
    "actionAngleStaeckelGrid",
    lambda P: aa.actionAngleStaeckelGrid(
        pot=MNa(P), delta=P["delta"], nE=12, npsi=12, nLz=15, c=True
    ),
    {**MNp, "delta": 0.4},
    act,
    diff_coords=["R", "vz"],
)
add_aa(
    "actionAngleAdiabaticGrid",
    lambda P: aa.actionAngleAdiabaticGrid(
        pot=MNa(P), gamma=P["gamma"], nR=12, nEz=12, nEr=20, nLz=20, c=True
    ),
    {**MNp, "gamma": 1.0},
    act,
    diff_coords=["R", "vz"],
)


def hinv(o, C):
    r = o.xvFreqs(C["J"], C["ang"])
    return {"x": r[0], "v": r[1], "O": r[2]}


add_aa(
    "actionAngleHarmonicInverse",
    lambda P: aa.actionAngleHarmonicInverse(omega=P["omega"]),
    {"omega": 1.3},
    hinv,
    coords={"J": 0.2, "ang": 0.7},
)


def iinv(o, C):
    r = o.xvFreqs(*[C[k] for k in ("jr", "lz", "jz", "ar", "ap", "az")])
    return {
        "R": r[0],
        "vR": r[1],
        "vT": r[2],
        "z": r[3],
        "vz": r[4],
        "Or": r[6],
    }


add_aa(
    "actionAngleIsochroneInverse",
    lambda P: aa.actionAngleIsochroneInverse(b=P["b"]),
    {"b": 1.2},
    iinv,
    coords={"jr": 0.1, "lz": 1.1, "jz": 0.05, "ar": 0.4, "ap": 0.5, "az": 0.6},
    diff_coords=["jr", "ar"],
)
add_aa(
    "actionAngleVerticalInverse",
    lambda P: aa.actionAngleVerticalInverse(
        pot=gp.KGPotential(K=P["K"], F=P["F"], D=0.5),
        Es=list(np.linspace(0.05, 1.0, 8)),
        setup_interp=True,
    ),
    {"K": 1.0, "F": 0.5},
    hinv,
    coords={"J": 0.1, "ang": 0.7},
)
add_aa(
    "estimateDeltaStaeckel",
    lambda P: MNa(P),
    dict(MNp),
    lambda p, C: {"delta": aa.estimateDeltaStaeckel(p, C["R"], C["z"])},
    coords={"R": 1.0, "z": 0.2},
    covers=(),
)
add_aa(
    "estimateBIsochrone",
    lambda P: MNa(P),
    dict(MNp),
    lambda p, C: {"b": aa.estimateBIsochrone(p, C["R"], C["z"])},
    coords={"R": 1.0, "z": 0.2},
    covers=(),
)


# Orbit
def _ic(C):
    v = [C[k] for k in ("R", "vR", "vT", "z", "vz", "phi")]
    xp = _xp(*v)
    return [float(x) for x in v] if xp is np else xp.stack([xp.asarray(x) for x in v])


def _ts(C):
    xp = _xp(C["R"])
    return np.linspace(0.0, 1.0, 11) if xp is np else xp.linspace(0.0, 1.0, 11)


def integ(method):
    def f(pot, C):
        xp = _xp(C["R"])
        m = method
        if m == "inbackend":
            if xp is np:
                m = "dop853_c"
            else:
                m = "diffrax" if "jax" in xp.__name__ else "torchdiffeq"
        o = Orbit(_ic(C))
        o.integrate(_ts(C), pot, method=m)
        return {"R1": o.R(1.0), "z1": o.z(1.0), "vR1": o.vR(1.0)}

    return f


for _m in (
    "inbackend",
    "dop853_c",
    "rk4_c",
    "rk6_c",
    "dopr54_c",
    "symplec4_c",
    "symplec6_c",
    "leapfrog_c",
    "ias15_c",
    "odeint",
    "dop853",
    "leapfrog",
):
    add_aa(
        f"Orbit.integrate[{_m}]",
        lambda P: MNa(P),
        dict(MNp),
        integ(_m),
        group="orbit",
        covers=(),
    )


def oan(typ, methods=("rperi", "rap", "zmax", "e", "jr", "jz", "Or", "wr")):
    def f(o_pot, C):
        pot, delta = o_pot
        o = Orbit(_ic(C))
        kw = dict(analytic=True, pot=pot, type=typ, use_physical=False)
        if typ == "staeckel":
            kw["delta"] = delta
        return {m: getattr(o, m)(**kw) for m in methods}

    return f


add_aa(
    "Orbit.analytic[staeckel]",
    lambda P: (MNa(P), P["delta"]),
    {**MNp, "delta": 0.4},
    oan("staeckel"),
    group="orbit",
    covers=(),
)
add_aa(
    "Orbit.analytic[adiabatic]",
    lambda P: (MNa(P), None),
    dict(MNp),
    oan("adiabatic", ("rperi", "rap", "zmax", "e", "jr", "jz")),
    group="orbit",
    covers=(),
)
add_aa(
    "Orbit.analytic[spherical]",
    lambda P: (gp.HernquistPotential(amp=4.0, a=P["a"]), None),
    {"a": 0.8},
    oan("spherical"),
    group="orbit",
    covers=(),
)
add_aa(
    "Orbit.E,Jacobi,Lz",
    lambda P: MNa(P),
    dict(MNp),
    lambda p, C: (
        lambda o: {"E": o.E(pot=p), "Jacobi": o.Jacobi(pot=p, OmegaP=0.3), "Lz": o.Lz()}
    )(Orbit(_ic(C))),
    group="orbit",
    covers=(),
)


# util.coords: every input is a parameter
def U(name, fn, params, idx=(0, 1, 2)):
    def out(P, C):
        r = fn(P)
        if isinstance(r, tuple):
            return {f"c{i}": r[i] for i in idx}
        return {f"c{i}": r.reshape(-1)[i] for i in idx}

    add_aa(name, lambda P: P, params, out, coords={}, group="util", covers=())


U(
    "coords.radec_to_lb",
    lambda P: gc.radec_to_lb(P["ra"], P["dec"]),
    {"ra": 1.0, "dec": 0.3},
    (0, 1),
)
U(
    "coords.radec_to_lb[degree]",
    lambda P: gc.radec_to_lb(P["ra"], P["dec"], degree=True),
    {"ra": 60.0, "dec": 20.0},
    (0, 1),
)
U(
    "coords.radec_to_lb[epoch=1950]",
    lambda P: gc.radec_to_lb(P["ra"], P["dec"], epoch=1950),
    {"ra": 1.0, "dec": 0.3},
    (0, 1),
)
U(
    "coords.lb_to_radec",
    lambda P: gc.lb_to_radec(P["l"], P["b"]),
    {"l": 1.0, "b": 0.3},
    (0, 1),
)
U(
    "coords.lbd_to_XYZ",
    lambda P: gc.lbd_to_XYZ(P["l"], P["b"], P["d"]),
    {"l": 1.0, "b": 0.3, "d": 2.0},
)
U(
    "coords.XYZ_to_lbd",
    lambda P: gc.XYZ_to_lbd(P["X"], P["Y"], P["Z"]),
    {"X": 1.0, "Y": 0.5, "Z": 0.3},
)
U(
    "coords.XYZ_to_galcenrect",
    lambda P: gc.XYZ_to_galcenrect(
        P["X"], P["Y"], P["Z"], Xsun=P["Xsun"], Zsun=P["Zsun"]
    ),
    {"X": 1.0, "Y": 0.5, "Z": 0.3, "Xsun": 1.0, "Zsun": 0.02},
)
U(
    "coords.galcenrect_to_XYZ",
    lambda P: gc.galcenrect_to_XYZ(
        P["X"], P["Y"], P["Z"], Xsun=P["Xsun"], Zsun=P["Zsun"]
    ),
    {"X": 1.0, "Y": 0.5, "Z": 0.3, "Xsun": 1.0, "Zsun": 0.02},
)
U(
    "coords.rect_to_cyl",
    lambda P: gc.rect_to_cyl(P["X"], P["Y"], P["Z"]),
    {"X": 1.0, "Y": 0.5, "Z": 0.3},
)
U(
    "coords.cyl_to_rect",
    lambda P: gc.cyl_to_rect(P["R"], P["phi"], P["Z"]),
    {"R": 1.0, "phi": 0.5, "Z": 0.3},
)
U(
    "coords.vxvyvz_to_galcencyl",
    lambda P: gc.vxvyvz_to_galcencyl(
        P["vx"],
        P["vy"],
        P["vz"],
        P["X"],
        0.5,
        0.3,
        vsun=[0.0, 1.0, 0.0],
        Xsun=1.0,
        Zsun=P["Zsun"],
    ),
    {"vx": 0.1, "vy": 0.2, "vz": 0.05, "X": 1.0, "Zsun": 0.02},
)
U(
    "coords.galcencyl_to_vxvyvz",
    lambda P: gc.galcencyl_to_vxvyvz(
        P["vR"],
        P["vT"],
        P["vz"],
        P["phi"],
        vsun=[0.0, 1.0, 0.0],
        Xsun=1.0,
        Zsun=P["Zsun"],
    ),
    {"vR": 0.1, "vT": 1.0, "vz": 0.05, "phi": 0.3, "Zsun": 0.02},
)
U(
    "coords.vrpmllpmbb_to_vxvyvz",
    lambda P: gc.vrpmllpmbb_to_vxvyvz(
        P["vr"], P["pmll"], P["pmbb"], P["l"], P["b"], P["d"]
    ),
    {"vr": 10.0, "pmll": 2.0, "pmbb": 1.0, "l": 1.0, "b": 0.3, "d": 2.0},
)
U(
    "coords.pmrapmdec_to_pmllpmbb",
    lambda P: gc.pmrapmdec_to_pmllpmbb(P["pmra"], P["pmdec"], P["ra"], P["dec"]),
    {"pmra": 2.0, "pmdec": 1.0, "ra": 1.0, "dec": 0.3},
    (0, 1),
)
U(
    "coords.Rz_to_coshucosv",
    lambda P: gc.Rz_to_coshucosv(P["R"], P["z"], delta=P["delta"]),
    {"R": 1.0, "z": 0.3, "delta": 0.5},
    (0, 1),
)
U(
    "coords.Rz_to_uv",
    lambda P: gc.Rz_to_uv(P["R"], P["z"], delta=P["delta"]),
    {"R": 1.0, "z": 0.3, "delta": 0.5},
    (0, 1),
)
U(
    "coords.Rz_to_lambdanu",
    lambda P: gc.Rz_to_lambdanu(P["R"], P["z"], ac=P["ac"], Delta=P["Delta"]),
    {"R": 1.0, "z": 0.3, "ac": 5.0, "Delta": 1.0},
    (0, 1),
)
U(
    "coords.lambdanu_to_Rz",
    lambda P: gc.lambdanu_to_Rz(P["l"], P["n"], ac=5.0, Delta=1.0),
    {"l": 2.09, "n": 0.0857},
    (0, 1),
)
U(
    "coords.cyl_to_spher",
    lambda P: gc.cyl_to_spher(P["R"], P["Z"], P["phi"]),
    {"R": 1.0, "Z": 0.3, "phi": 0.5},
)
U(
    "coords.sphergal_to_rectgal",
    lambda P: gc.sphergal_to_rectgal(
        P["l"], P["b"], P["d"], P["vr"], P["pmll"], P["pmbb"], degree=False
    ),
    {"l": 1.0, "b": 0.3, "d": 2.0, "vr": 10.0, "pmll": 2.0, "pmbb": 1.0},
)
_T = np.array([[0.4, 0.5, 0.766], [-0.8, 0.6, 0.0], [-0.46, -0.6, 0.643]])
U(
    "coords.radec_to_custom",
    lambda P: gc.radec_to_custom(P["ra"], P["dec"], T=_T),
    {"ra": 1.0, "dec": 0.3},
    (0, 1),
)
U(
    "coords.custom_to_radec",
    lambda P: gc.custom_to_radec(P["phi1"], P["phi2"], T=_T),
    {"phi1": 1.0, "phi2": 0.3},
    (0, 1),
)
U(
    "coords.pmrapmdec_to_custom",
    lambda P: gc.pmrapmdec_to_custom(P["pmra"], P["pmdec"], P["ra"], P["dec"], T=_T),
    {"pmra": 2.0, "pmdec": 1.0, "ra": 1.0, "dec": 0.3},
    (0, 1),
)
for _nm in (
    "dens_in_msolpc3",
    "dens_in_gevcc",
    "dens_in_meanmatterdens",
    "dens_in_criticaldens",
    "time_in_Gyr",
    "force_in_kmsMyr",
    "mass_in_msol",
    "mass_in_1010msol",
    "freq_in_Gyr",
    "surfdens_in_msolpc2",
    "velocity_in_kpcGyr",
    "force_in_pcMyr2",
    "freq_in_kmskpc",
    "force_in_2piGmsolpc2",
    "force_in_10m13kms2",
):
    add_aa(
        f"conversion.{_nm}",
        lambda P: P,
        {"vo": 220.0, "ro": 8.0},
        lambda P, C, nm=_nm: {"v": getattr(cv, nm)(P["vo"], P["ro"])},
        coords={},
        group="util",
        covers=(),
    )

SPECS_BY_NAME = {s["name"]: s for s in SPECS}
