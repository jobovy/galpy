###############################################################################
#   Backend-agnostic Gegenbauer (ultraspherical) polynomials C_n^alpha(x) for
#   0 <= n < N, via the standard three-term recurrence
#       C_0 = 1,  C_1 = 2 alpha x,
#       (n+1) C_{n+1} = 2(n+alpha) x C_n - (n+2 alpha-1) C_{n-1}.
#   This is the SCFPotential radial basis (galpy.potential.SCFPotential._C uses
#   the same recurrence with alpha = 2l + 3/2). Built with lists + xp.stack (no
#   in-place mutation), so it differentiates under jax and torch; the numpy path
#   reproduces SCF's existing recurrence value-for-value.
#
#   Under a jax trace the n-recurrence is rolled into a single ``lax.scan`` over n
#   (see ``_gegenbauer_scan_jax``): the eager Python loop would bake all N-1
#   recurrence steps into the user's jaxpr (and SCF stacks this over L orders, so
#   the radial basis unrolls to O(N*L) steps), whereas the scan traces the
#   recurrence body once. Eager numpy/torch/jax keep the Python loop unchanged
#   (byte-identical); the scan carries the same values at the same iteration
#   count, so the traced result matches the unrolled loop to the XLA fma-fusion
#   floor (the same ~1e-11 that jitting the unrolled loop itself shows vs the
#   eager per-op result -- the scan adds no error beyond that). Mirrors the landed
#   assoc_legendre lax.scan branch in the sibling file.
###############################################################################
from ..._namespaces import under_jax_trace


def gegenbauer(xp, N, alpha, x):
    """C_n^alpha(x) for 0 <= n < N, shape ``x.shape + (N,)``.

    N is a static int, alpha a scalar, x a backend array (or scalar).
    """
    x = xp.asarray(x) * 1.0
    # Traced: roll the recurrence into one lax.scan. Eager keeps the loop below.
    if under_jax_trace(x):
        return _gegenbauer_scan_jax(N, alpha, x)
    cols = [xp.ones_like(x)]  # C_0 = 1
    if N > 1:
        cnm1 = cols[0]
        cn = 2.0 * alpha * x  # C_1 = 2 alpha x
        cols.append(cn)
        for n in range(1, N - 1):
            cnp1 = (2.0 * (n + alpha) * x * cn - (n + 2.0 * alpha - 1.0) * cnm1) / (
                n + 1.0
            )
            cols.append(cnp1)
            cnm1, cn = cn, cnp1
    return xp.stack(cols, axis=-1)


def _gegenbauer_scan_jax(N, alpha, x):
    """Traced-only C_n^alpha(x) via a single ``lax.scan`` over n (carry the two
    previous columns C_{n-1}, C_{n-2}). Returns ``x.shape + (N,)`` (same as the
    eager list + ``xp.stack``).

    Each scanned step reproduces the eager arithmetic element-for-element: n==0
    injects C_0 = 1 and n==1 injects C_1 = 2 alpha x, while n>=2 uses the
    three-term recurrence C_n = (2(n-1+alpha) x C_{n-1} - (n+2 alpha-2) C_{n-2})/n
    (the eager C_{n+1} shifted by one). The n<2 denominator is guarded to 1 so the
    discarded recurrence branch stays finite (AD-safe); the ``where`` selects the
    base value there.
    """
    import jax
    import jax.numpy as jnp

    one = jnp.ones_like(x)
    c1 = 2.0 * alpha * x  # C_1

    def body(carry, n):
        cnm1, cnm2 = carry  # C_{n-1}, C_{n-2}
        nf = n * 1.0
        denom = jnp.where(n < 2, 1.0, nf)  # guard the n=0,1 division
        rec = (
            2.0 * (nf - 1.0 + alpha) * x * cnm1 - (nf + 2.0 * alpha - 2.0) * cnm2
        ) / denom
        cn = jnp.where(n == 0, one, jnp.where(n == 1, c1, rec))
        return (cn, cnm1), cn

    _, cols = jax.lax.scan(body, (one, one), jnp.arange(N))
    # cols: (N,) + x.shape -> x.shape + (N,) (matches the eager xp.stack(axis=-1)).
    return jnp.moveaxis(cols, 0, -1)
