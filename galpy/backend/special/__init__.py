###############################################################################
#   galpy.backend.special: native-preferring special-function router.
#
#   Each function dispatches, by the namespace of its array arguments, to the
#   backend's NATIVE implementation when present (scipy.special for numpy,
#   jax.scipy.special for jax, torch.special for torch) and otherwise to a
#   pure-backend, autodiff-friendly fallback in ``_fallback/``. This keeps the
#   numpy path byte-identical to scipy.special, lets jax/torch use their fast
#   native kernels, and makes galpy's special-function-backed methods (SCF,
#   Multipole, Einasto, PowerSphericalwCutoff, the disk potentials, spherical
#   DFs, ...) run and differentiate under every backend. As backends ship more
#   native functions the matching fallback is simply never reached (and a
#   capability test asserts native-vs-fallback agreement so it can be deleted).
###############################################################################
from ._router import (
    assoc_legendre,
    ellipe,
    ellipk,
    erf,
    erfc,
    gamma,
    gammainc,
    gammaincc,
    gammaln,
    gegenbauer,
    hyp1f1,
    hyp2f1,
    i0,
    i1,
    k0,
    k1,
    kn,
    xlogy,
)

__all__ = [
    "gamma",
    "gammaln",
    "gammainc",
    "gammaincc",
    "erf",
    "erfc",
    "i0",
    "i1",
    "xlogy",
    "hyp2f1",
    "hyp1f1",
    "ellipk",
    "ellipe",
    "k0",
    "k1",
    "kn",
    "assoc_legendre",
    "gegenbauer",
]
