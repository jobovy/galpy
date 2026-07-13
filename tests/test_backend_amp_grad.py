###############################################################################
# test_backend_amp_grad.py: autodiff w.r.t. the potential amplitude (d/damp)
# for potentials whose __init__ rescales self._amp.
#
# Many potentials adjust their amplitude in __init__ (e.g. self._amp /= 4 pi a^3)
# to convert a density normalization into the internal amplitude. When that
# rescaling is done *in place* (self._amp /= X), building the potential with a
# torch leaf tensor amp (requires_grad=True) raises
#     RuntimeError: a leaf Variable that requires grad is being used in an
#                   in-place operation
# and blocks amp-gradients entirely. Rewriting each rescaling out of place
# (self._amp = self._amp / (X), parentheses preserving the exact numpy value)
# unblocks jax/torch amp-gradients while staying byte-identical on numpy.
#
# The amplitude enters every force/potential as the outer factor
# (self._amp * self._Rforce(...)), so the amp-gradient equals the amp-free base
# force -- exactly reproduced by a finite difference. We assert AD == central FD
# (Richardson-extrapolated reference) for both jax and torch, plus a
# finite-difference-independent jax-vs-torch agreement check.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    CosmphiDiskPotential,
    EllipticalDiskPotential,
    FlattenedPowerPotential,
    MN3ExponentialDiskPotential,
    PerfectEllipsoidPotential,
    PowerSphericalPotential,
    PowerSphericalPotentialwCutoff,
    PowerTriaxialPotential,
    TriaxialGaussianPotential,
    TriaxialHernquistPotential,
    TriaxialJaffePotential,
    TriaxialNFWPotential,
    TwoPowerTriaxialPotential,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# Evaluation point: (R, z) for 3D potentials, (R, phi) for planar potentials.
_R0, _Z0, _PHI0 = 1.3, 0.4, 0.7


def _rforce_3d(pot):
    return pot.Rforce(_R0, _Z0, use_physical=False)


def _rforce_planar(pot):
    return pot.Rforce(_R0, phi=_PHI0, use_physical=False)


# (id, builder(amp) -> potential, force(pot) -> scalar, amp0). Each fixed file
# is represented; the four TwoPowerTriaxial-family rescalings (base class + the
# NFW/Hernquist/Jaffe subclasses) are covered by distinct entries. Ring and the
# Chandrasekhar dynamical-friction forces are intentionally excluded: their
# force evaluation is eager-only (a separate backend gap), so their amp-gradient
# is not yet meaningful here.
SPECS = [
    (
        "TriaxialGaussian",
        lambda a: TriaxialGaussianPotential(amp=a, sigma=0.9, b=0.8, c=0.7),
        _rforce_3d,
        1.1,
    ),
    (
        "MN3ExponentialDisk-sech",
        lambda a: MN3ExponentialDiskPotential(amp=a, hr=1.0, hz=0.3, sech=True),
        _rforce_3d,
        1.2,
    ),
    (
        "MN3ExponentialDisk-exp",
        lambda a: MN3ExponentialDiskPotential(amp=a, hr=1.0, hz=0.3, sech=False),
        _rforce_3d,
        1.2,
    ),
    (
        "FlattenedPower",
        lambda a: FlattenedPowerPotential(amp=a, alpha=0.6, q=0.9, r1=1.2),
        _rforce_3d,
        1.3,
    ),
    (
        "PowerSpherical",
        lambda a: PowerSphericalPotential(amp=a, alpha=1.5, r1=1.1),
        _rforce_3d,
        1.4,
    ),
    (
        "PowerSphericalwCutoff",
        lambda a: PowerSphericalPotentialwCutoff(amp=a, alpha=1.3, rc=1.4, r1=1.2),
        _rforce_3d,
        1.5,
    ),
    (
        "PowerTriaxial",
        lambda a: PowerTriaxialPotential(amp=a, alpha=1.7, r1=1.1, b=0.85, c=0.7),
        _rforce_3d,
        1.6,
    ),
    (
        "PerfectEllipsoid",
        lambda a: PerfectEllipsoidPotential(amp=a, a=1.3, b=0.9, c=0.75),
        _rforce_3d,
        1.7,
    ),
    (
        "TwoPowerTriaxial",
        lambda a: TwoPowerTriaxialPotential(
            amp=a, a=1.2, alpha=1.5, beta=3.5, b=0.9, c=0.7
        ),
        _rforce_3d,
        1.8,
    ),
    (
        "TriaxialNFW",
        lambda a: TriaxialNFWPotential(amp=a, a=1.2, b=0.9, c=0.7),
        _rforce_3d,
        1.9,
    ),
    (
        "TriaxialHernquist",
        lambda a: TriaxialHernquistPotential(amp=a, a=1.1, b=0.85, c=0.65),
        _rforce_3d,
        2.0,
    ),
    (
        "TriaxialJaffe",
        lambda a: TriaxialJaffePotential(amp=a, a=1.05, b=0.8, c=0.6),
        _rforce_3d,
        2.1,
    ),
    (
        "CosmphiDisk",
        lambda a: CosmphiDiskPotential(amp=a, m=2, p=1.0, phib=0.3, r1=1.2),
        _rforce_planar,
        1.1,
    ),
    (
        "EllipticalDisk",
        lambda a: EllipticalDiskPotential(amp=a, p=1.0, twophio=0.02, phib=0.4, r1=1.1),
        _rforce_planar,
        1.2,
    ),
]
SPEC_IDS = [s[0] for s in SPECS]

_EPS = 1e-4


def _fd_reference(builder, force, amp0):
    # Richardson-extrapolated central finite difference (O(eps^4)) on numpy.
    def fnp(a):
        return float(force(builder(a)))

    d1 = (fnp(amp0 + _EPS) - fnp(amp0 - _EPS)) / (2 * _EPS)
    d2 = (fnp(amp0 + _EPS / 2) - fnp(amp0 - _EPS / 2)) / _EPS
    return (4 * d2 - d1) / 3


def _ad_grad(backend_name, builder, force, amp0):
    if backend_name == "jax":
        return float(jax.grad(lambda a: force(builder(a)))(jnp.asarray(amp0)))
    at = torch.tensor(amp0, requires_grad=True)
    force(builder(at)).backward()
    return float(at.grad)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_amp_grad_vs_finite_difference(backend_name, spec):
    _, builder, force, amp0 = spec
    fd = _fd_reference(builder, force, amp0)
    ad = _ad_grad(backend_name, builder, force, amp0)
    # AD is exact; the limiting error is the finite-difference reference. The
    # gradient magnitudes here are all >~5e-3, so rtol dominates; rtol=1e-8
    # keeps a wide margin over the observed ~1e-12 relative agreement while
    # staying robust to cross-platform float rounding.
    numpy.testing.assert_allclose(ad, fd, rtol=1e-8, atol=1e-11)


@pytest.mark.skipif(
    "jax" not in BACKENDS or "torch" not in BACKENDS,
    reason="needs both jax and torch",
)
@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_amp_grad_jax_vs_torch(spec):
    # Finite-difference-independent: both backends compute the exact derivative
    # by autodiff, so they must agree to ~machine precision.
    _, builder, force, amp0 = spec
    g_jax = _ad_grad("jax", builder, force, amp0)
    g_torch = _ad_grad("torch", builder, force, amp0)
    numpy.testing.assert_allclose(g_jax, g_torch, rtol=1e-10, atol=1e-12)
